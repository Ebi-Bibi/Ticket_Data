"""Movie/cinema ticket forecast with MASE-aligned training and validation.

Forecast objective: use each movie/cinema pair's first three observed days to
predict its next seven daily ticket counts. Validation and training use the
competition's per-pair scale: max(mean(D1, D2, D3), 1).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


SEED = 2026
CALIBRATION_GRID = np.arange(0.40, 1.301, 0.01)
PAIR_KEYS = ["movie_title", "cinema_ids"]
HISTORY_VALUES = ["total_ticket", "total_show", "occupation_rate"]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "output" else SCRIPT_DIR
REQUIRED_FILES = [
    "train.csv", "test.csv", "test_history.csv", "movies.csv",
    "holidays.csv", "ticket_prices.csv", "sample_submission.csv",
]
INPUT_CANDIDATES = (
    PROJECT_DIR / "input",
    PROJECT_DIR / "data",
    PROJECT_DIR / "data" / "raw",
    SCRIPT_DIR / "data",
)
def resolve_input_dir():
    input_dir = next(
        (
            candidate
            for candidate in INPUT_CANDIDATES
            if all((candidate / name).is_file() for name in REQUIRED_FILES)
        ),
        None,
    )
    if input_dir is not None:
        return input_dir

    searched = "\n".join(
        str(path)
        for path in INPUT_CANDIDATES
    )
    raise FileNotFoundError(
        f"Required input CSVs were not found. Expected {REQUIRED_FILES}; searched:\n{searched}"
    )

OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_inputs(input_dir=None):
    input_dir = resolve_input_dir() if input_dir is None else Path(input_dir)
    train = pd.read_csv(input_dir / "train.csv", parse_dates=["date_show"])
    test = pd.read_csv(input_dir / "test.csv", parse_dates=["date_show"])
    history = pd.read_csv(input_dir / "test_history.csv", parse_dates=["date_show"])
    movies = pd.read_csv(input_dir / "movies.csv")
    holidays = pd.read_csv(input_dir / "holidays.csv", parse_dates=["date"])
    prices = pd.read_csv(input_dir / "ticket_prices.csv")
    sample = pd.read_csv(input_dir / "sample_submission.csv")
    return train, test, history, movies, holidays, prices, sample


def aggregate_daily_transactions(df):
    """Normalize transaction-like inputs to one row per movie/cinema/date."""
    df = df.copy()
    df["date_show"] = pd.to_datetime(df["date_show"])
    group_cols = PAIR_KEYS + ["date_show"]
    if not df.duplicated(group_cols).any():
        return df.sort_values(group_cols).reset_index(drop=True)

    agg_spec = {}
    if "city_name" in df:
        agg_spec["city_name"] = "first"
    if "total_ticket" in df:
        agg_spec["total_ticket"] = "sum"
    if "total_show" in df:
        agg_spec["total_show"] = "sum"

    if "occupation_rate" in df and "total_show" in df:
        df["_occupancy_weighted_sum"] = (
            df["occupation_rate"].fillna(0) * df["total_show"].fillna(0)
        )
        agg_spec["_occupancy_weighted_sum"] = "sum"
        daily = df.groupby(group_cols, as_index=False).agg(agg_spec)
        daily["occupation_rate"] = np.where(
            daily["total_show"].gt(0),
            daily["_occupancy_weighted_sum"] / daily["total_show"],
            0,
        )
        daily = daily.drop(columns="_occupancy_weighted_sum")
    elif "occupation_rate" in df:
        agg_spec["occupation_rate"] = "mean"
        daily = df.groupby(group_cols, as_index=False).agg(agg_spec)
    else:
        daily = df.groupby(group_cols, as_index=False).agg(agg_spec)

    return daily.sort_values(group_cols).reset_index(drop=True)


def mase(y_true, y_pred, scale):
    """Competition MASE: mean rowwise absolute error divided by pair scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    scale = np.maximum(np.asarray(scale, dtype=float), 1.0)
    return float(np.mean(np.abs(y_true - y_pred) / scale))


def make_complete_training_rows(train):
    """Build every contiguous ten-day D1-D3 + D4-D10 training example."""
    df = train.sort_values(PAIR_KEYS + ["date_show"]).reset_index(drop=True).copy()
    grouped = df.groupby(PAIR_KEYS, sort=False)
    date_at_offset = [grouped["date_show"].shift(-offset) for offset in range(10)]
    consecutive = np.ones(len(df), dtype=bool)
    start_dates = df["date_show"]
    for offset, shifted_dates in enumerate(date_at_offset):
        consecutive &= shifted_dates.eq(start_dates + pd.to_timedelta(offset, unit="D")).to_numpy()

    base = df.loc[consecutive, PAIR_KEYS + ["city_name", "date_show"]].copy()
    base = base.rename(columns={"date_show": "history_start"}).reset_index(drop=True)
    # Compute target shifts on the full series first, then retain only valid
    # contiguous ten-day starts. Shifting after filtering would skip days.
    for offset in range(3):
        base[f"d{offset + 1}_ticket"] = grouped["total_ticket"].shift(-offset).loc[consecutive].to_numpy()
        for source_col, feature_name in (("total_show", "shows"), ("occupation_rate", "occupancy")):
            base[f"d{offset + 1}_{feature_name}"] = grouped[source_col].shift(-offset).loc[consecutive].to_numpy()
    for offset in range(7):
        base[f"future_{offset + 1}"] = grouped["total_ticket"].shift(-(offset + 3)).loc[consecutive].to_numpy()
    base["window_end"] = base["history_start"] + pd.Timedelta(days=9)

    samples = []
    shared = PAIR_KEYS + [
        "city_name", "history_start", "window_end", "d1_ticket", "d2_ticket", "d3_ticket",
        "d1_shows", "d2_shows", "d3_shows", "d1_occupancy", "d2_occupancy", "d3_occupancy",
    ]
    for horizon in range(1, 8):
        frame = base[shared].copy()
        frame["horizon"] = horizon
        frame["target"] = base[f"future_{horizon}"].to_numpy()
        frame["target_date"] = frame["history_start"] + pd.to_timedelta(2 + horizon, unit="D")
        samples.append(frame)
    return pd.concat(samples, ignore_index=True)


def make_missing_aware_training_rows(train):
    """Reindex each pair by calendar date; an absent sale record means 0 tickets.

    Each series is extended up to 9 zero days past its last record (never past
    the last train date) so windows include movies leaving the screen. A window
    is kept only if D4-D10 contains at least one real record, matching test.
    """
    train_end = train["date_show"].max()
    windows = []
    for (movie, cinema), group in train.groupby(PAIR_KEYS, sort=False):
        group = group.sort_values("date_show").set_index("date_show")
        series_end = min(group.index.max() + pd.Timedelta(days=9), train_end)
        dates = pd.date_range(group.index.min(), series_end, freq="D")
        daily = group.reindex(dates)
        present = daily["total_ticket"].notna().to_numpy()
        if len(dates) < 10:
            continue
        ticket = daily["total_ticket"].fillna(0).to_numpy(dtype=float)
        shows = daily["total_show"].fillna(0).to_numpy(dtype=float)
        occupancy = daily["occupation_rate"].fillna(0).to_numpy(dtype=float)
        n_starts = len(dates) - 9
        future_present = np.lib.stride_tricks.sliding_window_view(present[3:], 7)
        starts = np.flatnonzero(future_present[:n_starts].any(axis=1))
        if len(starts) == 0:
            continue

        base = pd.DataFrame({
            "movie_title": movie,
            "cinema_ids": cinema,
            "city_name": group["city_name"].iloc[0],
            "history_start": dates.take(starts),
            "window_end": dates.take(starts + 9),
        })
        for day in range(3):
            base[f"d{day + 1}_ticket"] = ticket[starts + day]
            base[f"d{day + 1}_missing"] = (~present[starts + day]).astype(int)
            base[f"d{day + 1}_shows"] = shows[starts + day]
            base[f"d{day + 1}_occupancy"] = occupancy[starts + day]
        base["history_present_days"] = (
            present[starts].astype(int)
            + present[starts + 1].astype(int)
            + present[starts + 2].astype(int)
        )

        for horizon in range(1, 8):
            frame = base.copy()
            target_offset = horizon + 2
            frame["horizon"] = horizon
            frame["target"] = ticket[starts + target_offset]
            frame["target_date"] = dates.take(starts + target_offset)
            windows.append(frame)
    if not windows:
        raise ValueError("No missing-aware ten-day training windows were built.")
    return pd.concat(windows, ignore_index=True)


def make_test_rows(test, history):
    """Attach each test forecast's own D1-D3 history and derive its horizon."""
    frame = test.copy()
    group_start = frame.groupby(PAIR_KEYS)["date_show"].transform("min")
    group_end = frame.groupby(PAIR_KEYS)["date_show"].transform("max")
    frame["forecast_start"] = group_start
    frame["horizon"] = (frame["date_show"] - group_start).dt.days + 1
    frame["history_start"] = group_start - pd.Timedelta(days=3)
    frame["window_end"] = group_end
    counts = frame.groupby(PAIR_KEYS)["date_show"].transform("nunique")
    if not counts.eq(7).all() or not frame["horizon"].between(1, 7).all():
        raise ValueError("Each test movie/cinema pair must have seven forecast days.")
    expected = group_start + pd.to_timedelta(frame["horizon"] - 1, unit="D")
    if not frame["date_show"].eq(expected).all():
        raise ValueError("Test forecast days must be consecutive for each pair.")

    lookup = history[PAIR_KEYS + ["date_show"] + HISTORY_VALUES].copy()
    for day_index, days_back in enumerate((3, 2, 1), start=1):
        date_col = f"d{day_index}_date"
        ticket_col = f"d{day_index}_ticket"
        frame[date_col] = group_start - pd.Timedelta(days=days_back)
        right = lookup[[*PAIR_KEYS, "date_show"] + HISTORY_VALUES].rename(
            columns={
                "date_show": date_col,
                "total_ticket": ticket_col,
                "total_show": f"d{day_index}_shows",
                "occupation_rate": f"d{day_index}_occupancy",
            }
        )
        frame = frame.merge(right, on=PAIR_KEYS + [date_col], how="left", validate="many_to_one")
        frame[f"d{day_index}_missing"] = frame[ticket_col].isna().astype(int)
    frame[["d1_ticket", "d2_ticket", "d3_ticket"]] = frame[
        ["d1_ticket", "d2_ticket", "d3_ticket"]
    ].fillna(0)
    history_extra_cols = [f"d{i}_{name}" for i in range(1, 4) for name in ("shows", "occupancy")]
    frame[history_extra_cols] = frame[history_extra_cols].fillna(0)
    frame["target_date"] = frame["date_show"]
    return frame


def add_movie_history_features(samples, source):
    """As-of movie level ticket strength; observations on/after D1 are excluded."""
    source = source.dropna(subset=["total_ticket"]).copy()
    daily = source.groupby(["movie_title", "date_show"], as_index=False).agg(
        ticket_sum=("total_ticket", "sum"),
        ticket_count=("total_ticket", "count"),
        ticket_sq_sum=("total_ticket", lambda values: np.square(values).sum()),
    ).sort_values(["date_show", "movie_title"])
    daily["cum_sum"] = daily.groupby("movie_title")["ticket_sum"].cumsum() - daily["ticket_sum"]
    daily["cum_count"] = daily.groupby("movie_title")["ticket_count"].cumsum() - daily["ticket_count"]
    daily["cum_sq_sum"] = daily.groupby("movie_title")["ticket_sq_sum"].cumsum() - daily["ticket_sq_sum"]

    left = samples[["movie_title", "history_start"]].copy()
    left["_row_order"] = np.arange(len(left))
    left = left.sort_values(["history_start", "movie_title"])
    right = daily[["movie_title", "date_show", "cum_sum", "cum_count", "cum_sq_sum"]].copy()
    right = right.sort_values(["date_show", "movie_title"])
    history_features = pd.merge_asof(
        left,
        right,
        left_on="history_start",
        right_on="date_show",
        by="movie_title",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row_order")
    count = history_features["cum_count"].replace(0, np.nan)
    mean = history_features["cum_sum"] / count
    variance = history_features["cum_sq_sum"] / count - mean.pow(2)
    result = samples.copy()
    result["movie_hist_mean"] = mean.to_numpy()
    result["movie_hist_std"] = np.sqrt(variance.clip(lower=0)).to_numpy()
    result["movie_hist_count"] = history_features["cum_count"].fillna(0).to_numpy()
    result[["movie_hist_mean", "movie_hist_std"]] = result[
        ["movie_hist_mean", "movie_hist_std"]
    ].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result


def add_pair_and_cinema_history(samples, source):
    """Add strictly past pair/cinema demand summaries, excluding the D1 date."""
    source = source.dropna(subset=["total_ticket"]).copy()
    result = samples.copy()
    result["_row_order"] = np.arange(len(result))

    for name, keys in (("pair", PAIR_KEYS), ("cinema", ["cinema_ids"])):
        daily = source.groupby(keys + ["date_show"], as_index=False).agg(
            day_ticket_sum=("total_ticket", "sum"),
            day_ticket_count=("total_ticket", "count"),
        )
        daily = daily.sort_values(keys + ["date_show"])
        grouped = daily.groupby(keys, sort=False)
        daily["cum_sum"] = grouped["day_ticket_sum"].cumsum() - daily["day_ticket_sum"]
        daily["cum_count"] = grouped["day_ticket_count"].cumsum() - daily["day_ticket_count"]
        daily["daily_mean"] = daily["day_ticket_sum"] / daily["day_ticket_count"]
        shifted_mean = daily.groupby(keys, sort=False)["daily_mean"].shift(1)
        daily[f"{name}_last7_mean"] = shifted_mean.groupby(
            [daily[key] for key in keys], sort=False
        ).transform(lambda values: values.rolling(7, min_periods=1).mean())
        daily["prior_date"] = daily.groupby(keys, sort=False)["date_show"].shift(1)

        left = result[keys + ["history_start", "_row_order"]].copy()
        left = left.sort_values(["history_start"] + keys)
        right = daily[keys + ["date_show", "cum_sum", "cum_count", f"{name}_last7_mean", "prior_date"]]
        right = right.sort_values(["date_show"] + keys)
        past = pd.merge_asof(
            left,
            right,
            left_on="history_start",
            right_on="date_show",
            by=keys,
            direction="backward",
            allow_exact_matches=False,
        ).sort_values("_row_order")
        mean = past["cum_sum"] / past["cum_count"].replace(0, np.nan)
        result[f"{name}_hist_mean"] = mean.to_numpy()
        result[f"{name}_hist_count"] = past["cum_count"].fillna(0).to_numpy()
        result[f"{name}_last7_mean"] = past[f"{name}_last7_mean"].fillna(0).to_numpy()
        result[f"{name}_days_since_last"] = (
            pd.to_datetime(result["history_start"]).to_numpy()
            - pd.to_datetime(past["prior_date"]).to_numpy()
        ) / np.timedelta64(1, "D")
        result[f"{name}_days_since_last"] = pd.Series(
            result[f"{name}_days_since_last"], index=result.index
        ).replace([np.inf, -np.inf], np.nan).fillna(999)

    hist_cols = [
        "pair_hist_mean", "pair_hist_count", "pair_last7_mean", "pair_days_since_last",
        "cinema_hist_mean", "cinema_hist_count", "cinema_last7_mean", "cinema_days_since_last",
    ]
    result[hist_cols] = result[hist_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)


def add_features(samples, history_source, movies, holidays, prices):
    frame = samples.copy()
    history_cols = ["d1_ticket", "d2_ticket", "d3_ticket"]
    for index in range(1, 4):
        missing_col = f"d{index}_missing"
        if missing_col not in frame:
            frame[missing_col] = 0
    if "history_present_days" not in frame:
        frame["history_present_days"] = 3 - frame[[f"d{i}_missing" for i in range(1, 4)]].sum(axis=1)
    frame[history_cols] = frame[history_cols].fillna(0)
    frame["scale"] = frame[history_cols].mean(axis=1).clip(lower=1)
    for index, col in enumerate(history_cols, start=1):
        frame[f"d{index}_ratio"] = frame[col] / frame["scale"]
        for suffix in ("shows", "occupancy"):
            value_col = f"d{index}_{suffix}"
            if value_col not in frame:
                frame[value_col] = 0.0
    frame["shows_mean"] = frame[[f"d{i}_shows" for i in range(1, 4)]].mean(axis=1)
    frame["shows_max"] = frame[[f"d{i}_shows" for i in range(1, 4)]].max(axis=1)
    frame["shows_trend"] = frame["d3_shows"] - frame["d1_shows"]
    frame["occupancy_mean"] = frame[[f"d{i}_occupancy" for i in range(1, 4)]].mean(axis=1)
    frame["occupancy_max"] = frame[[f"d{i}_occupancy" for i in range(1, 4)]].max(axis=1)
    frame["occupancy_trend"] = frame["d3_occupancy"] - frame["d1_occupancy"]
    frame["history_mean"] = frame[history_cols].mean(axis=1)
    frame["history_std"] = frame[history_cols].std(axis=1).fillna(0)
    frame["history_min"] = frame[history_cols].min(axis=1)
    frame["history_max"] = frame[history_cols].max(axis=1)
    frame["growth_2"] = frame["d2_ticket"] / frame["d1_ticket"].clip(lower=1)
    frame["growth_3"] = frame["d3_ticket"] / frame["d2_ticket"].clip(lower=1)
    frame["overall_growth"] = frame["d3_ticket"] / frame["d1_ticket"].clip(lower=1)
    frame["diff_2"] = frame["d2_ticket"] - frame["d1_ticket"]
    frame["diff_3"] = frame["d3_ticket"] - frame["d2_ticket"]
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    dates = frame["target_date"]
    frame["day_of_week"] = dates.dt.dayofweek
    frame["day_of_month"] = dates.dt.day
    frame["month"] = dates.dt.month
    frame["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    frame["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    frame["is_friday"] = (dates.dt.dayofweek == 4).astype(int)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)
    frame["week_sin"] = np.sin(2 * np.pi * frame["week_of_year"] / 52)
    frame["week_cos"] = np.cos(2 * np.pi * frame["week_of_year"] / 52)

    frame = add_movie_history_features(frame, history_source)
    frame = add_pair_and_cinema_history(frame, history_source)
    movie_meta = movies[["original_title", "age_rating", "genre"]].rename(
        columns={"original_title": "movie_title"}
    )
    frame = frame.merge(movie_meta.drop_duplicates("movie_title"), on="movie_title", how="left", validate="many_to_one")
    frame["pair_key"] = frame["movie_title"].astype(str) + "::" + frame["cinema_ids"].astype(str)
    holiday_features = holidays[["date", "day_tipe", "holiday_tipe"]].rename(columns={"date": "target_date"})
    frame = frame.merge(holiday_features.drop_duplicates("target_date"), on="target_date", how="left", validate="many_to_one")
    frame["day_tipe"] = frame["day_tipe"].fillna("no_holiday")
    frame["holiday_tipe"] = frame["holiday_tipe"].fillna("no_holiday")
    frame["price_day"] = np.select(
        [frame["day_of_week"].eq(4), frame["day_of_week"].ge(5)],
        ["Friday", "Weekend"],
        default="Weekday",
    )
    price_features = prices.rename(columns={"ceil": "ticket_price"})
    frame = frame.merge(
        price_features.drop_duplicates(["city_name", "price_day"]),
        on=["city_name", "price_day"],
        how="left",
        validate="many_to_one",
    )
    city_median = frame.groupby("city_name")["ticket_price"].transform("median")
    frame["ticket_price"] = frame["ticket_price"].fillna(city_median).fillna(frame["ticket_price"].median())
    return frame


def attach_movie_release_dates(samples, release_dates):
    result = samples.merge(release_dates, on="movie_title", how="left", validate="many_to_one")
    if result["movie_release_date"].isna().any():
        missing = result.loc[result["movie_release_date"].isna(), "movie_title"].nunique()
        raise ValueError(f"Missing release date for {missing} movies in training samples")
    return result


V3_FEATURES = [
    "d1_ratio", "d2_ratio", "d3_ratio", "history_mean", "history_std",
    "history_min", "history_max", "growth_2", "growth_3", "overall_growth",
    "diff_2", "diff_3", "horizon", "day_of_month", "is_weekend", "is_friday",
    "dow_sin", "dow_cos", "movie_hist_mean", "movie_hist_std", "movie_hist_count",
]
V4_FEATURES = V3_FEATURES + [
    "pair_hist_mean", "pair_hist_count", "pair_last7_mean", "pair_days_since_last",
    "cinema_hist_mean", "cinema_hist_count", "cinema_last7_mean", "cinema_days_since_last",
]
V5_FEATURES = V4_FEATURES + [
    "d1_missing", "d2_missing", "d3_missing", "history_present_days",
    "shows_mean", "shows_max", "shows_trend", "occupancy_mean",
    "occupancy_max", "occupancy_trend", "ticket_price", "is_carryover",
]


def make_model(variant):
    if variant == "modelv4":
        # Modelv4's selected, regularized LightGBM settings are the benchmark.
        return LGBMRegressor(
            objective="regression_l1", n_estimators=700, learning_rate=0.03,
            num_leaves=15, max_depth=-1, min_child_samples=40, subsample=0.85,
            subsample_freq=1, colsample_bytree=0.85, reg_lambda=5.0, random_state=SEED,
            n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True,
        )
    # The history-enriched model gets enough capacity for cinema/pair patterns,
    # while regularization limits memorization of unstable low-history groups.
    return LGBMRegressor(
        objective="regression_l1", n_estimators=900, learning_rate=0.025,
        num_leaves=31, max_depth=-1, min_child_samples=60, subsample=0.85,
        subsample_freq=1, colsample_bytree=0.85, reg_lambda=8.0, random_state=SEED,
        n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True,
    )


def main():
    input_dir = resolve_input_dir()
    train, test, test_history, movies, holidays, prices, sample = read_inputs(input_dir)
    print("INPUT DIRECTORY:", input_dir)
    print("TRAIN / TEST / HISTORY:", train.shape, test.shape, test_history.shape)
    print("TRAIN DATE RANGE:", train["date_show"].min().date(), "to", train["date_show"].max().date())
    print("TICKET SUMMARY:")
    print(train["total_ticket"].describe(percentiles=[0.5, 0.9, 0.99]))
    print("DUPLICATE TRAIN PAIR-DATES:", train.duplicated(PAIR_KEYS + ["date_show"]).sum())
    print("DUPLICATE HISTORY PAIR-DATES:", test_history.duplicated(PAIR_KEYS + ["date_show"]).sum())
    if len(test) != len(sample):
        raise ValueError(f"test.csv rows ({len(test)}) must match sample_submission rows ({len(sample)})")

    train = aggregate_daily_transactions(train)
    test_history = aggregate_daily_transactions(test_history)
    print("DAILY TRAIN / HISTORY:", train.shape, test_history.shape)

    release_dates = (
        train.groupby("movie_title", as_index=False)["date_show"]
        .min()
        .rename(columns={"date_show": "movie_release_date"})
    )
    train_start = train["date_show"].min()

    complete_samples = make_complete_training_rows(train)
    complete_samples = add_features(complete_samples, train, movies, holidays, prices)
    complete_samples = attach_movie_release_dates(complete_samples, release_dates)
    complete_samples["is_carryover"] = complete_samples["movie_release_date"].eq(train_start).astype(int)
    samples = make_missing_aware_training_rows(train)
    samples = add_features(samples, train, movies, holidays, prices)
    samples = attach_movie_release_dates(samples, release_dates)
    samples["is_carryover"] = samples["movie_release_date"].eq(train_start).astype(int)
    print("MODEL V4 COMPLETE-RECORD WINDOWS:", complete_samples.shape)
    print("MODEL V5 ZERO-FILLED WINDOWS:", samples.shape)
    print("MODEL V5 ZERO FUTURE TARGETS:", int(samples["target"].eq(0).sum()))

    test_samples = make_test_rows(test, test_history)
    history_source = pd.concat([train, test_history], ignore_index=True)
    test_samples = add_features(test_samples, history_source, movies, holidays, prices)
    test_history_start = test_history["date_show"].min()
    train_movies = set(train["movie_title"].unique())
    test_samples["is_carryover"] = (
        test_samples["movie_title"].isin(train_movies)
        | test_samples["history_start"].eq(test_history_start)
    ).astype(int)
    print("TEST HORIZON COUNTS:", test_samples["horizon"].value_counts().sort_index().to_dict())

    categorical_v4 = ["age_rating", "genre", "cinema_ids", "city_name"]
    categorical_v5 = categorical_v4 + ["day_tipe", "holiday_tipe", "price_day"]
    category_maps = {}
    for column in categorical_v5:
        values = pd.concat(
            [complete_samples[column], samples[column], test_samples[column]], ignore_index=True
        ).fillna("__MISSING__").astype(str)
        category_maps[column] = pd.Index(values.unique())
        complete_samples[column] = pd.Categorical(complete_samples[column].fillna("__MISSING__").astype(str), categories=category_maps[column])
        samples[column] = pd.Categorical(samples[column].fillna("__MISSING__").astype(str), categories=category_maps[column])
        test_samples[column] = pd.Categorical(test_samples[column].fillna("__MISSING__").astype(str), categories=category_maps[column])
    feature_cols = {
        "modelv4": V4_FEATURES + categorical_v4,
        "modelv5": V5_FEATURES + categorical_v5,
    }
    categorical_cols = {"modelv4": categorical_v4, "modelv5": categorical_v5}
    for cols in feature_cols.values():
        samples[cols] = samples[cols].replace([np.inf, -np.inf], np.nan)
        complete_samples[cols] = complete_samples.reindex(columns=cols).replace([np.inf, -np.inf], np.nan)
        test_samples[cols] = test_samples[cols].replace([np.inf, -np.inf], np.nan)

    release_block_starts = pd.to_datetime([
        "2025-05-01", "2025-05-29", "2025-06-26",
        "2025-07-24", "2025-08-21", "2025-09-18",
    ])
    release_block_days = 28
    model_names = ["modelv4", "modelv5"]
    oof_predictions = {name: [] for name in model_names}
    actuals, horizons = [], []
    walk_forward_errors = {name: 0.0 for name in model_names}
    walk_forward_count = 0
    eligible_samples = samples.copy()
    eligible_complete_samples = complete_samples.copy()
    for block_start in release_block_starts:
        block_end = block_start + pd.Timedelta(days=release_block_days)
        validation = eligible_samples[
            eligible_samples["movie_release_date"].between(
                block_start,
                block_end - pd.Timedelta(days=1),
            )
        ].copy()
        fit_rows_v4 = eligible_complete_samples[
            eligible_complete_samples["movie_release_date"].lt(block_start)
        ].copy()
        fit_rows_v5 = eligible_samples[
            eligible_samples["movie_release_date"].lt(block_start)
        ].copy()
        if validation.empty or fit_rows_v4.empty or fit_rows_v5.empty:
            print(
                f"SKIP RELEASE VALIDATION {block_start.date()}.."
                f"{(block_end - pd.Timedelta(days=1)).date()}: no train or validation windows"
            )
            continue
        actual_scaled = validation["target"].to_numpy() / validation["scale"].to_numpy()
        horizon_values = validation["horizon"].to_numpy()
        fold_preds = {}
        prior_actuals = np.asarray(actuals, dtype=float)
        prior_horizons = np.asarray(horizons, dtype=int)
        for name in model_names:
            fit_rows = fit_rows_v4 if name == "modelv4" else fit_rows_v5
            model = make_model(name)
            model.fit(
                fit_rows[feature_cols[name]],
                fit_rows["target"].to_numpy() / fit_rows["scale"].to_numpy(),
                categorical_feature=categorical_cols[name],
            )
            fold_preds[name] = np.maximum(model.predict(validation[feature_cols[name]]), 0)

            # One calibration factor per horizon, learned only from earlier folds.
            factors = np.ones(8, dtype=float)
            if len(prior_actuals):
                grid = CALIBRATION_GRID
                for horizon in range(1, 8):
                    prior_mask = prior_horizons == horizon
                    if prior_mask.any():
                        scores = [
                            np.mean(np.abs(prior_actuals[prior_mask] - factor * np.asarray(oof_predictions[name])[prior_mask]))
                            for factor in grid
                        ]
                        factors[horizon] = grid[int(np.argmin(scores))]
            calibrated_fold_pred = fold_preds[name] * factors[horizon_values]
            walk_forward_errors[name] += float(np.abs(actual_scaled - calibrated_fold_pred).sum())

        walk_forward_count += len(validation)
        for name in model_names:
            oof_predictions[name].extend(fold_preds[name].tolist())
        actuals.extend(actual_scaled.tolist())
        horizons.extend(horizon_values.tolist())
        print(
            f"RELEASE VALIDATION {block_start.date()}.."
            f"{(block_end - pd.Timedelta(days=1)).date()} "
            f"movies={validation['movie_title'].nunique():,} rows={len(validation):,} "
            f"modelv4 MASE={np.mean(np.abs(actual_scaled - fold_preds['modelv4'])):.6f} "
            f"modelv5 MASE={np.mean(np.abs(actual_scaled - fold_preds['modelv5'])):.6f}"
        )

    if not actuals:
        raise ValueError("No temporal validation folds were produced.")
    y_scaled = np.asarray(actuals)
    horizon_array = np.asarray(horizons, dtype=int)
    final_factors = {}
    full_grid = CALIBRATION_GRID
    for name in model_names:
        oof_pred = np.asarray(oof_predictions[name])
        raw_score = float(np.mean(np.abs(y_scaled - oof_pred)))
        final_factors[name] = np.ones(8, dtype=float)
        for horizon in range(1, 8):
            mask = horizon_array == horizon
            scores = [np.mean(np.abs(y_scaled[mask] - factor * oof_pred[mask])) for factor in full_grid]
            final_factors[name][horizon] = full_grid[int(np.argmin(scores))]
        calibrated_score = float(np.mean(np.abs(y_scaled - oof_pred * final_factors[name][horizon_array])))
        walk_forward_score = walk_forward_errors[name] / walk_forward_count
        print(
            f"POOLED {name}: raw MASE={raw_score:.6f}; "
            f"calibrated MASE={calibrated_score:.6f}; "
            f"walk-forward calibrated MASE={walk_forward_score:.6f}"
        )
    winner = min(model_names, key=lambda name: walk_forward_errors[name] / walk_forward_count)
    print(f"SELECTED MODEL: {winner}; horizon factors={np.round(final_factors[winner][1:], 3).tolist()}")

    # Final fit uses only the validation-selected training-window recipe.
    final_train = complete_samples if winner == "modelv4" else samples
    all_y = final_train["target"].to_numpy() / final_train["scale"].to_numpy()
    final_model = make_model(winner)
    final_model.fit(final_train[feature_cols[winner]], all_y, categorical_feature=categorical_cols[winner])
    pred_scaled_raw = np.maximum(final_model.predict(test_samples[feature_cols[winner]]), 0)
    pred_scaled = pred_scaled_raw * final_factors[winner][test_samples["horizon"].to_numpy()]
    submission = pd.DataFrame({
        "id": test_samples["id"].to_numpy(),
        "total_ticket": pred_scaled * test_samples["scale"].to_numpy(),
    })
    if submission["id"].duplicated().any():
        raise ValueError("Submission IDs must be unique before reordering to sample_submission.csv")
    submission = (
        submission
        .set_index("id")
        .reindex(sample["id"])
        .reset_index()
    )

    if submission.columns.tolist() != sample.columns.tolist():
        raise ValueError(f"Unexpected submission columns: {submission.columns.tolist()}")
    if not submission["id"].equals(sample["id"]):
        raise ValueError("Submission IDs/order do not match sample_submission.csv")
    if not np.isfinite(submission["total_ticket"]).all() or (submission["total_ticket"] < 0).any():
        raise ValueError("Submission predictions must be finite and nonnegative")

    output_path = OUTPUT_DIR / "submission_modelv8.csv"
    submission.to_csv(output_path, index=False)
    saved = pd.read_csv(output_path)

    print("\n" + "=" * 68)
    print("SUBMISSION CHECK")
    print("=" * 68)
    print("Shape:", submission.shape)
    print(submission.head(10).to_string(index=False))
    print("Prediction summary:\n", submission["total_ticket"].describe())
    print("Modelv5 mean prediction:", f"{submission['total_ticket'].mean():.3f}")
    for version in ("modelv3", "modelv4"):
        previous_submission_path = OUTPUT_DIR / f"submission_{version}.csv"
        if previous_submission_path.is_file():
            previous = pd.read_csv(previous_submission_path)
            if previous["id"].equals(submission["id"]):
                delta = submission["total_ticket"] - previous["total_ticket"]
                print(f"{version} mean prediction:", f"{previous['total_ticket'].mean():.3f}")
                print(
                    f"Rows lower/higher than {version}:",
                    int((delta < 0).sum()), "/", int((delta > 0).sum()),
                )
    print("This comparison is prediction size only; MASE determines model quality.")
    print("Missing:", submission["total_ticket"].isna().sum())
    print("Negative:", (submission["total_ticket"] < 0).sum())

    print("\n" + "=" * 68)
    print("FINAL SUBMISSION SANITY CHECK")
    print("=" * 68)
    print("Saved to:", output_path)
    print("Shape:", saved.shape)
    print("Columns:", saved.columns.tolist())
    print("Unique IDs / expected:", saved["id"].nunique(), "/", len(sample))
    print("IDs match sample order:", saved["id"].equals(sample["id"]))
    print("NaN / negative / zero:", saved["total_ticket"].isna().sum(), ",", (saved["total_ticket"] < 0).sum(), ",", (saved["total_ticket"] == 0).sum())
    print("Quantiles:\n", saved["total_ticket"].quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]))


if __name__ == "__main__":
    main()
