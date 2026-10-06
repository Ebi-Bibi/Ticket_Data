"""Local validation harness that mimics how the competition test set is built.

Test construction (verified in step 0): D1 is the film's wide-release date
(its first date in test_history, shared by all its pairs), and a pair is in
test iff it has a record on D3. Targets D4-D10 are zero-filled.

Step 1 picks a wide-release D1 definition for train films that matches test.
Step 2 builds release-anchored episodes (rule A) and scores baselines and the
modelv4 recipe (the one behind leaderboard 0.58309) per validation block.
Step 3 (only if step 2 lands in range) compares a release-episode candidate.

Model code is imported from modelv5.py unchanged.
"""

import numpy as np
import pandas as pd

import modelv5 as m5

PAIR_KEYS = m5.PAIR_KEYS
HORIZONS = range(1, 8)
# Horizon factors printed by modelv5 when it selected modelv4 (D4..D10).
V5_LOG_FACTORS = np.array([0.96, 0.94, 0.94, 0.92, 0.92, 0.91, 0.93])
VALIDATION_BLOCKS = [
    ("Jul", "2025-07-01", "2025-07-31"),
    ("Aug", "2025-08-01", "2025-08-31"),
    ("Sep 1-21", "2025-09-01", "2025-09-21"),
]
MASE_RANGE = (0.50, 0.68)
CATEGORICAL_V4 = ["age_rating", "genre", "cinema_ids", "city_name"]
FEATURES_V4 = m5.V4_FEATURES + CATEGORICAL_V4
# Movie and pair history only ever see the same film's records before D1.
# Calendar-season features and cinema recency/volume fall outside the train
# range in test (Oct-Mar, months after train ends), so they are excluded too.
OUT_OF_RANGE_FEATURES = {
    "month_sin", "month_cos", "week_sin", "week_cos", "movie_title", "pair_key",
    "cinema_last7_mean", "cinema_days_since_last", "cinema_hist_count",
}
FEATURES_CANDIDATE = [
    f for f in m5.V4_FEATURES
    if not f.startswith(("movie_hist", "pair_")) and f not in OUT_OF_RANGE_FEATURES
] + CATEGORICAL_V4
CHOSEN_D1_DEFINITION = (1, 0.5, 3)  # min shows, share of peak, sustained days
# Wide-release definitions: (cinema basis, min shows per cinema-day, share of peak, sustained days).
D1_DEFINITIONS = [
    (f"cinemas>={frac:.0%} peak x{days}d" + ("" if min_shows == 1 else f", shows>={min_shows}"), min_shows, frac, days)
    for min_shows in (1, 3)
    for frac in (0.3, 0.5, 0.7)
    for days in (1, 3)
]


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def with_day_index(df, d1):
    """Attach film-level D1 and k = days since D1 + 1 (k=1 is D1)."""
    out = df.merge(d1.rename("D1"), left_on="movie_title", right_index=True)
    out["k"] = (out["date_show"] - out["D1"]).dt.days + 1
    return out


def release_profile(df, d1):
    """The four test-matching statistics for a given film -> D1 mapping."""
    early = with_day_index(df[df["movie_title"].isin(d1.index)], d1)
    early = early[early["k"].between(1, 3)]
    per_pair = early.groupby(PAIR_KEYS)["k"].agg(["size", lambda k: (k == 3).any()])
    per_pair.columns = ["records", "has_d3"]
    weekday = d1.dt.dayofweek.map({3: "Thu", 2: "Wed", 4: "Fri"}).fillna("other")
    shares = weekday.value_counts(normalize=True).reindex(["Thu", "Wed", "Fri", "other"], fill_value=0)
    return {
        "films": len(d1),
        "d3_share": per_pair["has_d3"].mean(),
        "Thu": shares["Thu"], "Wed": shares["Wed"], "Fri": shares["Fri"], "other": shares["other"],
        "records_per_pair": per_pair["records"].mean(),
        "d3_pairs_per_film": per_pair["has_d3"].sum() / len(d1),
    }


def test_reference(test, history):
    section("STEP 0: test construction and reference statistics")
    first_test = test.groupby(PAIR_KEYS)["date_show"].min()
    hist_d1 = history.groupby("movie_title")["date_show"].min()
    pair_d1 = (first_test - pd.Timedelta(days=3)).rename("D1").reset_index()
    pair_d1 = pair_d1.merge(hist_d1.rename("hist_first"), left_on="movie_title", right_index=True)
    hist = with_day_index(history, hist_d1)
    has_d3 = hist.groupby(PAIR_KEYS)["k"].apply(lambda k: (k == 3).any())
    in_test = has_d3.index.isin(first_test.index)
    print(f"test pair D1 == film's first test_history date: {pair_d1['D1'].eq(pair_d1['hist_first']).mean():.2%}")
    print(f"history pairs in test with D3 record: {has_d3[in_test].mean():.2%}; "
          f"not in test with D3 record: {int(has_d3[~in_test].sum())}")
    reference = release_profile(history, hist_d1)
    print("test reference:", {k: round(float(v), 4) for k, v in reference.items()})
    return reference


def cinema_counts(train, min_shows):
    counted = train[train["total_show"].fillna(0) >= min_shows]
    return counted.groupby(["movie_title", "date_show"])["cinema_ids"].nunique()


def wide_release_dates(train, min_shows, frac, days):
    """First date whose cinema count >= frac * peak for `days` consecutive days."""
    counts = cinema_counts(train, min_shows)
    first_dates = train.groupby("movie_title")["date_show"].min()
    last_date = train["date_show"].max()
    out = {}
    for movie, series in counts.groupby(level="movie_title"):
        series = series.droplevel("movie_title")
        series = series.reindex(pd.date_range(first_dates[movie], last_date, freq="D"), fill_value=0)
        values = series.to_numpy()
        ok = values >= frac * values.max()
        # Near the end of train fewer than `days` days remain; require what exists.
        run = np.array([ok[i:i + days].all() for i in range(len(ok))])
        if run.any():
            out[movie] = series.index[int(np.argmax(run))]
    return pd.Series(out, dtype="datetime64[ns]")


def eligible_release_dates(d1, train, test_movies):
    """Drop test films, films already showing on the first train day, and D10 past train."""
    first_dates = train.groupby("movie_title")["date_show"].min()
    train_start, train_end = train["date_show"].min(), train["date_show"].max()
    keep = (
        ~d1.index.isin(test_movies)
        & first_dates.reindex(d1.index).gt(train_start).to_numpy()
        & (d1 + pd.Timedelta(days=9) <= train_end).to_numpy()
    )
    return d1[keep]


def distance(profile, reference):
    """Sum of normalized gaps to test; weights are the tolerance for each stat."""
    weekday_gap = 0.5 * sum(abs(profile[d] - reference[d]) for d in ("Thu", "Wed", "Fri", "other"))
    return (
        abs(profile["d3_share"] - reference["d3_share"]) / 0.02
        + weekday_gap / 0.05
        + abs(profile["records_per_pair"] - reference["records_per_pair"]) / 0.05
        + abs(np.log(profile["d3_pairs_per_film"] / reference["d3_pairs_per_film"])) / 0.25
    )


def step1(train, test, reference, override=None):
    section("STEP 1: wide-release D1 definitions for train films")
    test_movies = set(test["movie_title"])
    first_dates = train.groupby("movie_title")["date_show"].min()
    print(f"train films: {len(first_dates)}; in test: {len(test_movies & set(first_dates.index))}; "
          f"showing on {train['date_show'].min().date()}: {int(first_dates.eq(train['date_show'].min()).sum())}")
    rows, mappings = [], {}
    baseline = eligible_release_dates(first_dates, train, test_movies)
    candidates = [("first date (old)", baseline)] + [
        (name, eligible_release_dates(wide_release_dates(train, min_shows, frac, days), train, test_movies))
        for name, min_shows, frac, days in D1_DEFINITIONS
    ]
    for name, d1 in candidates:
        profile = release_profile(train, d1)
        profile["preview_films"] = int((d1 > first_dates.reindex(d1.index)).sum())
        profile["distance"] = distance(profile, reference)
        rows.append({"definition": name, **profile})
        mappings[name] = d1
    rows.append({"definition": "TEST", **reference})
    table = pd.DataFrame(rows).set_index("definition")
    print(table.round(3).to_string())
    chosen = table.drop(index=["TEST", "first date (old)"])["distance"].idxmin()
    if override is not None:
        chosen = override
    print(f"\nCHOSEN D1 DEFINITION: {chosen}{'' if override else ' (lowest distance)'}; "
          f"films kept: {len(mappings[chosen])}")
    return mappings[chosen]


def preview_free_source(train, d1):
    """Remove every record a film has before its D1; test never shows previews."""
    start = train["movie_title"].map(d1)
    return train[~(start.notna() & train["date_show"].lt(start))].reset_index(drop=True)


def build_episodes(train, d1):
    """One episode per pair with a record on D3 (rule A); D1-D3 and D4-D10 zero-filled."""
    tr = with_day_index(train[train["movie_title"].isin(d1.index)], d1)
    tr = tr[tr["k"].between(1, 10)]
    pairs = tr[tr["k"].eq(3)].groupby(PAIR_KEYS).agg(city_name=("city_name", "first"), D1=("D1", "first"))
    wide = {
        name: tr.pivot_table(index=PAIR_KEYS, columns="k", values=col, aggfunc="sum")
        .reindex(index=pairs.index, columns=range(1, 11))
        for name, col in (("ticket", "total_ticket"), ("shows", "total_show"), ("occupancy", "occupation_rate"))
    }
    base = pairs.reset_index()
    base["history_start"] = base["D1"]
    base["window_end"] = base["D1"] + pd.Timedelta(days=9)
    for day in (1, 2, 3):
        base[f"d{day}_ticket"] = wide["ticket"][day].fillna(0).to_numpy()
        base[f"d{day}_shows"] = wide["shows"][day].fillna(0).to_numpy()
        base[f"d{day}_occupancy"] = wide["occupancy"][day].fillna(0).to_numpy()
        base[f"d{day}_missing"] = wide["ticket"][day].isna().astype(int).to_numpy()
    frames = []
    for horizon in HORIZONS:
        frame = base.copy()
        frame["horizon"] = horizon
        frame["target"] = wide["ticket"][horizon + 3].fillna(0).to_numpy()
        frame["target_date"] = frame["D1"] + pd.Timedelta(days=horizon + 2)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def scaled_error(target, pred, scale):
    return np.abs(np.asarray(target) - np.asarray(pred)) / np.asarray(scale)


def by_horizon(frame, columns):
    table = pd.DataFrame({name: values.groupby(frame["horizon"]).mean() for name, values in columns.items()}).T
    table.columns = [f"D{h + 3}" for h in table.columns]
    return table


def block_masks(episodes):
    for label, start, end in VALIDATION_BLOCKS:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        yield label, start, episodes["D1"].between(start, end)


def describe_episodes(episodes):
    section("STEP 2a: release-anchored episodes (rule A, preview-free)")
    in_blocks = np.zeros(len(episodes), dtype=bool)
    for label, _, mask in block_masks(episodes):
        in_blocks |= mask.to_numpy()
        print(f"block {label:8s} films={episodes.loc[mask, 'movie_title'].nunique():3d} "
              f"pairs={int(mask.sum()) // 7:,}")
    print(f"all episodes: films={episodes['movie_title'].nunique()}, "
          f"pairs={len(episodes) // 7:,}, rows={len(episodes):,}; "
          f"in validation blocks: films={episodes.loc[in_blocks, 'movie_title'].nunique()}, pairs={int(in_blocks.sum()) // 7:,}")
    ratio = episodes["target"] / episodes["scale"]
    print(by_horizon(episodes, {
        "zero_share": episodes["target"].eq(0),
        "mean_target/scale": ratio,
    }).round(3).to_string())


def step2_baselines(episodes):
    section("STEP 2b: baselines on validation blocks (median ratio from fit episodes only)")
    rows, pooled = [], {"all zero": [], "pred = scale": [], "pred = D3": [], "median ratio (fit)": []}
    for label, start, mask in block_masks(episodes):
        block = episodes[mask]
        fit = episodes[episodes["window_end"] < start]
        medians = (fit["target"] / fit["scale"]).groupby(fit["horizon"]).median()
        preds = {
            "all zero": np.zeros(len(block)),
            "pred = scale": block["scale"],
            "pred = D3": block["d3_ticket"],
            "median ratio (fit)": block["horizon"].map(medians) * block["scale"],
        }
        row = {"block": label, "fit_films": fit["movie_title"].nunique()}
        for name, pred in preds.items():
            err = scaled_error(block["target"], pred, block["scale"])
            pooled[name].append(err)
            row[name] = err.mean()
        rows.append(row)
    rows.append({"block": "POOLED", **{name: np.concatenate(errs).mean() for name, errs in pooled.items()}})
    print(pd.DataFrame(rows).set_index("block").round(4).to_string())


def encode_categories(frames):
    for column in CATEGORICAL_V4:
        values = pd.concat([f[column] for f in frames]).fillna("__MISSING__").astype(str)
        categories = pd.Index(values.unique())
        for f in frames:
            f[column] = pd.Categorical(f[column].fillna("__MISSING__").astype(str), categories=categories)
    for f in frames:
        f[m5.V4_FEATURES] = f[m5.V4_FEATURES].replace([np.inf, -np.inf], np.nan)


def fit_predict(fit, block, features):
    model = m5.make_model("modelv4")
    model.fit(fit[features], fit["target"] / fit["scale"], categorical_feature=CATEGORICAL_V4)
    return np.maximum(model.predict(block[features]), 0) * block["scale"].to_numpy()


def run_modelv4(complete, episodes):
    section("STEP 2c: modelv4 recipe (complete-record windows, full-train features)")
    blocks = []
    for label, start, mask in block_masks(episodes):
        block = episodes[mask].copy()
        fit = complete[complete["window_end"] < start]
        block["block"] = label
        block["pred_v4"] = fit_predict(fit, block, FEATURES_V4)
        block["pred_v4_cal"] = block["pred_v4"] * V5_LOG_FACTORS[block["horizon"].to_numpy() - 1]
        blocks.append(block)
        print(f"block {label:8s} fit_windows={len(fit) // 7:,} films={block['movie_title'].nunique()} "
              f"pairs={len(block) // 7:,} MASE raw={scaled_error(block['target'], block['pred_v4'], block['scale']).mean():.4f} "
              f"calibrated={scaled_error(block['target'], block['pred_v4_cal'], block['scale']).mean():.4f}")
    result = pd.concat(blocks, ignore_index=True)
    raw = scaled_error(result["target"], result["pred_v4"], result["scale"])
    cal = scaled_error(result["target"], result["pred_v4_cal"], result["scale"])
    print(f"POOLED modelv4 MASE raw={raw.mean():.4f} calibrated={cal.mean():.4f} (rows={len(result):,})")
    print(by_horizon(result, {
        "MASE_raw": pd.Series(raw), "MASE_cal": pd.Series(cal),
        "pred/scale": result["pred_v4"] / result["scale"],
        "target/scale": result["target"] / result["scale"],
    }).round(3).to_string())
    return result, raw.mean()


def step3(result, episodes):
    section("STEP 3: release-episode candidate vs modelv4 recipe")
    blocks = []
    for label, start, _ in block_masks(episodes):
        block = result[result["block"].eq(label)].copy()
        fit = episodes[episodes["window_end"] < start]
        block["pred_cand"] = fit_predict(fit, block, FEATURES_CANDIDATE)
        blocks.append(block)
        print(f"block {label:8s} candidate fit: films={fit['movie_title'].nunique()} pairs={len(fit) // 7:,}")
    result = pd.concat(blocks, ignore_index=True)
    result["pred_blend"] = 0.5 * (result["pred_v4"] + result["pred_cand"])
    models = {"modelv4": "pred_v4", "candidate": "pred_cand", "blend 50/50": "pred_blend"}
    rows = []
    for label in [b[0] for b in VALIDATION_BLOCKS] + ["POOLED"]:
        part = result if label == "POOLED" else result[result["block"].eq(label)]
        rows.append({"block": label, **{
            name: scaled_error(part["target"], part[col], part["scale"]).mean() for name, col in models.items()
        }})
    print(pd.DataFrame(rows).set_index("block").round(4).to_string())
    print("\nzero-prediction share:", {name: round(float(result[col].eq(0).mean()), 4) for name, col in models.items()},
          "| zero-target share:", round(float(result["target"].eq(0).mean()), 4))
    print(by_horizon(result, {
        "target/scale": result["target"] / result["scale"],
        **{f"{name} pred/scale": result[col] / result["scale"] for name, col in models.items()},
    }).round(3).to_string())


def main(d1_definition=None):
    train, test, history, movies, holidays, prices, _ = m5.read_inputs()
    train = m5.aggregate_daily_transactions(train)
    history = m5.aggregate_daily_transactions(history)
    reference = test_reference(test, history)
    d1 = step1(train, test, reference, override=d1_definition)

    source = preview_free_source(train, d1)
    episodes = build_episodes(train, d1)
    episodes = m5.add_features(episodes, source, movies, holidays, prices)
    describe_episodes(episodes)
    step2_baselines(episodes)

    complete = m5.make_complete_training_rows(train)
    complete = m5.add_features(complete, train, movies, holidays, prices)
    encode_categories([complete, episodes])
    result, pooled = run_modelv4(complete, episodes)

    low, high = MASE_RANGE
    if not low <= pooled <= high:
        print(f"\nSTOP: pooled modelv4 MASE {pooled:.4f} is outside {low:.2f}-{high:.2f}; step 3 skipped.")
        return
    step3(result, episodes)


if __name__ == "__main__":
    import sys

    # Optional: a definition name from the step 1 table, e.g. "cinemas>=30% peak x3d, shows>=3".
    main(sys.argv[1] if len(sys.argv) > 1 else None)
