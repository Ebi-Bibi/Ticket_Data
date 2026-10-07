"""Experiments on model v10 (+A+B); harness.py and harness_v10.py are unchanged.

feature_test(): C (pair show/occupancy in D1-D3) and D (cinema behaviour from
past episodes whose D10 is before D1). Neither beat v10.
tuning_test(): error breakdown of v10 validation predictions, then LightGBM
tuning with v10 features fixed. No setting beat v10.
main(): E (cinema/city weekday pattern before D1) and F (market context of
films released in the 6 days up to D1), v10 features and parameters.
"""

import numpy as np
import pandas as pd

import harness as h
import harness_v10 as hv
import modelv5 as m5

FEATURES_C = (
    [f"c_shows_d{d}" for d in (1, 2, 3)] + [f"c_occ_d{d}" for d in (1, 2, 3)]
    + [f"c_tps_d{d}" for d in (1, 2, 3)] + ["c_show_trend"]
    + [f"d{d}_missing" for d in (1, 2, 3)] + ["c_occ_vs_film"]
)
FEATURES_D = ["cin_ratio_mean", "cin_zero_late", "cin_all_zero"]
MIN_PAST_EPISODES = 5
SEEDS = [2026, 1, 2, 3, 4]
BASE = hv.VARIANTS["+A+B"]
VARIANTS = {
    "v10": BASE,
    "+C": BASE + FEATURES_C,
    "+D": BASE + FEATURES_D,
    "+C+D": BASE + FEATURES_C + FEATURES_D,
}


def film_mean_occupancy(records, d1):
    early = h.with_day_index(records[records["movie_title"].isin(d1.index)], d1)
    return early[early["k"].between(1, 3)].groupby("movie_title")["occupation_rate"].mean()


def add_show_features(frame, film_occupancy):
    out = frame.copy()
    for d in (1, 2, 3):
        missing = out[f"d{d}_missing"].eq(1)
        out[f"c_shows_d{d}"] = out[f"d{d}_shows"]
        out[f"c_occ_d{d}"] = out[f"d{d}_occupancy"]
        out[f"c_tps_d{d}"] = (out[f"d{d}_ticket"] / out[f"d{d}_shows"].where(out[f"d{d}_shows"] > 0)).where(~missing)
    out["c_show_trend"] = out["d3_shows"] - out["d1_shows"]
    occ = out[[f"d{d}_occupancy" for d in (1, 2, 3)]].where(out[[f"d{d}_missing" for d in (1, 2, 3)]].eq(0).to_numpy())
    out["c_occ_vs_film"] = occ.mean(axis=1) / out["movie_title"].map(film_occupancy)
    return out


def episode_summary(episodes):
    """One row per past episode: cinema, D1, D10 and its target behaviour."""
    ratio = (episodes["target"] / episodes["scale"]).clip(upper=5)
    keys = [episodes[k] for k in h.PAIR_KEYS]
    late = episodes["horizon"].ge(5)
    return pd.DataFrame({
        "ratio": ratio.groupby(keys).mean(),
        "zero_late": episodes["target"].eq(0).astype(float).where(late).groupby(keys).mean(),
        "all_zero": episodes["target"].eq(0).groupby(keys).all().astype(float),
        "D1": episodes.groupby(keys)["history_start"].first(),
    }).reset_index().assign(window_end=lambda f: f["D1"] + pd.Timedelta(days=9))


def add_cinema_features(frame, past):
    """As-of per-cinema means over past episodes with D10 strictly before D1."""
    past = past.sort_values("window_end").copy()
    grouped = past.groupby("cinema_ids")
    for col in ("ratio", "zero_late", "all_zero"):
        past[f"cum_{col}"] = grouped[col].cumsum()
    past["cum_n"] = grouped.cumcount() + 1
    query = frame[["cinema_ids", "history_start"]].drop_duplicates()
    query = query.assign(D1=pd.to_datetime(query["history_start"])).sort_values("D1")
    merged = pd.merge_asof(
        query, past[["cinema_ids", "window_end", "cum_ratio", "cum_zero_late", "cum_all_zero", "cum_n"]],
        left_on="D1", right_on="window_end", by="cinema_ids", allow_exact_matches=False,
    )
    enough = merged["cum_n"].ge(MIN_PAST_EPISODES)
    for col, name in (("ratio", "cin_ratio_mean"), ("zero_late", "cin_zero_late"), ("all_zero", "cin_all_zero")):
        merged[name] = (merged[f"cum_{col}"] / merged["cum_n"]).where(enough)
    return frame.merge(merged[["cinema_ids", "history_start"] + FEATURES_D], on=["cinema_ids", "history_start"], how="left")


def prepare(train, test, history, movies, holidays, prices, with_test=False):
    min_shows, frac, days = h.CHOSEN_D1_DEFINITION
    d1 = h.eligible_release_dates(h.wide_release_dates(train, min_shows, frac, days), train, set(test["movie_title"]))
    episodes, source = hv.prepare_train(train, test, movies, holidays, prices)
    past = episode_summary(episodes)
    episodes = add_cinema_features(add_show_features(episodes, film_mean_occupancy(source, d1)), past)
    if not with_test:
        return episodes, None
    rows = hv.prepare_test(test, history, source, movies, holidays, prices)
    rows = add_show_features(rows, film_mean_occupancy(history, history.groupby("movie_title")["date_show"].min()))
    rows = add_cinema_features(rows, past)
    return episodes, rows


def clean(frames):
    hv.clean(frames)
    for f in frames:
        f[FEATURES_C + FEATURES_D] = f[FEATURES_C + FEATURES_D].replace([np.inf, -np.inf], np.nan)


def fit_predict(fit, block, features, seeds=(m5.SEED,), params=None):
    preds = []
    for seed in seeds:
        model = m5.make_model("modelv4").set_params(random_state=seed, **(params or {}))
        model.fit(fit[features], fit["target"] / fit["scale"], categorical_feature=h.CATEGORICAL_V4)
        preds.append(np.maximum(model.predict(block[features]), 0))
    return np.mean(preds, axis=0) * block["scale"].to_numpy()


def validation_predictions(episodes, features, seeds=(m5.SEED,), params=None):
    blocks = []
    for label, start, mask in h.block_masks(episodes):
        block = episodes[mask].copy()
        fit = episodes[episodes["window_end"] < start]
        block["block"] = label
        block["pred"] = fit_predict(fit, block, features, seeds, params)
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True)
    result["error"] = h.scaled_error(result["target"], result["pred"], result["scale"])
    return result


def score(episodes, features, seeds=(m5.SEED,), params=None):
    result = validation_predictions(episodes, features, seeds, params)
    per_block = result.groupby("block", sort=False)["error"].mean().to_dict()
    return {**per_block, "POOLED": result["error"].mean()}


def feature_test():
    train, test, history, movies, holidays, prices, _ = m5.read_inputs()
    train = m5.aggregate_daily_transactions(train)
    episodes, _ = prepare(train, test, history, movies, holidays, prices)
    clean([episodes])

    table = pd.DataFrame({name: score(episodes, feats) for name, feats in VARIANTS.items()}).T
    best = table.drop(index="v10")["POOLED"].idxmin()
    table.loc[f"{best} x5 seeds"] = score(episodes, VARIANTS[best], SEEDS)
    print(table.round(4).to_string())

    blocks = [b[0] for b in h.VALIDATION_BLOCKS]
    top = table.drop(index="v10")["POOLED"].idxmin()
    gain = table.loc["v10", "POOLED"] - table.loc[top, "POOLED"]
    wins = int((table.loc[top, blocks] < table.loc["v10", blocks]).sum())
    print(f"best={top} gain={gain:.4f} block_wins={wins}/3 -> {'PASS' if gain >= 0.005 and wins >= 2 else 'FAIL'}")


# Changes against the v10 LightGBM settings (modelv4 recipe).
TUNING = {
    "leaves 7": {"num_leaves": 7},
    "leaves 31": {"num_leaves": 31},
    "min_child 100": {"min_child_samples": 100},
    "sub .8 col .6": {"subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.6},
    "lr/2 trees x2": {"learning_rate": 0.015, "n_estimators": 1400},
}


def breakdown(result):
    """Row share, MASE and share of total error per group."""
    total = result["error"].sum()
    by_pair = result.groupby(h.PAIR_KEYS, observed=True)["target"].transform(lambda t: t.eq(0).all())
    groupings = {
        "horizon": result["horizon"] + 3,
        "scale": pd.cut(result["scale"], [0, 1, 5, 20, 100, np.inf], labels=["1", "1-5", "5-20", "20-100", ">100"]),
        "target": np.where(result["target"].eq(0), "target 0", "target >0"),
        "pair": np.where(by_pair, "all-zero pair", "other pair"),
    }
    for name, key in groupings.items():
        table = result.groupby(key, observed=True).agg(rows=("error", "size"), MASE=("error", "mean"), err=("error", "sum"))
        table["row_share"] = table.pop("rows") / len(result)
        table["err_share"] = table.pop("err") / total
        if name == "target":
            table["pred/scale"] = (result["pred"] / result["scale"]).groupby(key).mean()
        print(f"\n[{name}]\n" + table.round(4).to_string())


def tuning_test():
    train, test, _, movies, holidays, prices, _ = m5.read_inputs()
    train = m5.aggregate_daily_transactions(train)
    episodes, _ = hv.prepare_train(train, test, movies, holidays, prices)
    hv.clean([episodes])

    result = validation_predictions(episodes, BASE)
    print(f"v10 pooled MASE {result['error'].mean():.4f}")
    breakdown(result)

    rows = {"v10": {**result.groupby("block", sort=False)["error"].mean().to_dict(), "POOLED": result["error"].mean()}}
    for name, params in TUNING.items():
        rows[name] = score(episodes, BASE, params=params)
    table = pd.DataFrame(rows).T
    improved = [n for n in TUNING if table.loc[n, "POOLED"] < table.loc["v10", "POOLED"]]
    # num_leaves 7 and 31 conflict; keep whichever of them scored better.
    leaves = [n for n in improved if n.startswith("leaves")]
    if len(leaves) == 2:
        improved.remove(max(leaves, key=lambda n: table.loc[n, "POOLED"]))
    combined = {k: v for n in improved for k, v in TUNING[n].items()}
    if len(improved) > 1:
        table.loc["combined: " + " + ".join(improved)] = score(episodes, BASE, params=combined)
    print("\n" + table.round(4).to_string())

    blocks = [b[0] for b in h.VALIDATION_BLOCKS]
    top = table.drop(index="v10")["POOLED"].idxmin()
    gain = table.loc["v10", "POOLED"] - table.loc[top, "POOLED"]
    wins = int((table.loc[top, blocks] < table.loc["v10", blocks]).sum())
    print(f"best={top} gain={gain:.4f} block_wins={wins}/3 -> {'PASS' if gain >= 0.005 and wins >= 2 else 'FAIL'}")
    print("params:", combined if top.startswith("combined") else TUNING.get(top))


FEATURES_E = ["cinema_dow_ratio", "city_dow_ratio"]
FEATURES_F = ["market_films", "market_log_ticket", "market_share"]
MIN_PATTERN_DAYS = 28
MARKET_WINDOW_DAYS = 6


def weekday_cumulative(train, key):
    """Per key and date: running ticket sums and record counts per weekday."""
    records = train.dropna(subset=["total_ticket"])
    daily = records.groupby([key, "date_show"], observed=True)["total_ticket"].agg(["sum", "count"]).reset_index()
    dow = daily["date_show"].dt.dayofweek
    cols = []
    for d in range(7):
        daily[f"s{d}"] = daily["sum"].where(dow.eq(d), 0)
        daily[f"n{d}"] = daily["count"].where(dow.eq(d), 0)
        cols += [f"s{d}", f"n{d}"]
    daily["days"] = 1
    cols.append("days")
    daily[cols] = daily.groupby(key, observed=True)[cols].cumsum()
    return daily.sort_values("date_show")[[key, "date_show"] + cols]


def add_weekday_pattern(frame, cumulative, key, name):
    """Pattern on the target weekday over the mean pattern on D1-D3 weekdays."""
    query = frame[[key, "history_start"]].drop_duplicates().sort_values("history_start")
    merged = pd.merge_asof(query, cumulative, left_on="history_start", right_on="date_show",
                           by=key, allow_exact_matches=False)
    pattern = np.stack([merged[f"s{d}"] / merged[f"n{d}"].where(merged[f"n{d}"] > 0) for d in range(7)], axis=1)
    pattern[~merged["days"].ge(MIN_PATTERN_DAYS).to_numpy()] = np.nan
    rows = frame[[key, "history_start"]].merge(
        merged[[key, "history_start"]].reset_index(), on=[key, "history_start"], how="left"
    )["index"].to_numpy()
    d1_dow = frame["history_start"].dt.dayofweek.to_numpy()
    target = pattern[rows, frame["target_date"].dt.dayofweek.to_numpy()]
    early = np.mean(np.stack([pattern[rows, (d1_dow + i) % 7] for i in range(3)], axis=1), axis=1)
    out = frame.copy()
    out[name] = target / early
    return out


def release_totals(records, d1):
    """Film D1 and its D1-D3 ticket total over all cinemas."""
    early = h.with_day_index(records[records["movie_title"].isin(d1.index)], d1)
    totals = early[early["k"].between(1, 3)].groupby("movie_title")["total_ticket"].sum()
    return pd.DataFrame({"D1": d1, "total": totals.reindex(d1.index).fillna(0)})


def add_market_context(frame, films):
    """Other films with D1 in [D1 - 6 days, D1]; never films released later."""
    own = frame[["movie_title", "history_start", "film_total"]].drop_duplicates(["movie_title", "history_start"])
    stats = []
    for title, d1, own_total in own.itertuples(index=False):
        others = films[films["D1"].between(d1 - pd.Timedelta(days=MARKET_WINDOW_DAYS), d1) & (films.index != title)]
        other_total = others["total"].sum()
        stats.append((title, d1, len(others), np.log1p(other_total), own_total / max(own_total + other_total, 1)))
    stats = pd.DataFrame(stats, columns=["movie_title", "history_start"] + FEATURES_F)
    return frame.merge(stats, on=["movie_title", "history_start"], how="left")


def add_ef(frame, train, films):
    for key, name in (("cinema_ids", "cinema_dow_ratio"), ("city_name", "city_dow_ratio")):
        frame = add_weekday_pattern(frame, weekday_cumulative(train, key), key, name)
    return add_market_context(frame, films)


def prepare_ef(train, test, history, movies, holidays, prices, with_test=False):
    min_shows, frac, days = h.CHOSEN_D1_DEFINITION
    episodes, source = hv.prepare_train(train, test, movies, holidays, prices)
    train_films = release_totals(train, h.wide_release_dates(train, min_shows, frac, days))
    episodes = add_ef(episodes, train, train_films)
    if not with_test:
        return episodes, None
    rows = hv.prepare_test(test, history, source, movies, holidays, prices)
    rows["history_start"] = pd.to_datetime(rows["history_start"])
    test_films = release_totals(history, history.groupby("movie_title")["date_show"].min())
    return episodes, add_ef(rows, train, test_films)


def clean_ef(frames):
    hv.clean(frames)
    for f in frames:
        f[FEATURES_E + FEATURES_F] = f[FEATURES_E + FEATURES_F].replace([np.inf, -np.inf], np.nan)


EF_VARIANTS = {
    "v10": BASE,
    "+E": BASE + FEATURES_E,
    "+F": BASE + FEATURES_F,
    "+E+F": BASE + FEATURES_E + FEATURES_F,
}


def main():
    train, test, history, movies, holidays, prices, _ = m5.read_inputs()
    train = m5.aggregate_daily_transactions(train)
    episodes, _ = prepare_ef(train, test, history, movies, holidays, prices)
    clean_ef([episodes])
    print("NaN share:", episodes[FEATURES_E + FEATURES_F].isna().mean().round(3).to_dict())

    rows = {}
    for name, features in EF_VARIANTS.items():
        result = validation_predictions(episodes, features)
        by_h = result.groupby("horizon")["error"].mean()
        rows[name] = {**result.groupby("block", sort=False)["error"].mean().to_dict(),
                      "POOLED": result["error"].mean(), "D4": by_h[1], "D5": by_h[2]}
    table = pd.DataFrame(rows).T
    print(table.round(4).to_string())

    blocks = [b[0] for b in h.VALIDATION_BLOCKS]
    top = table.drop(index="v10")["POOLED"].idxmin()
    gain = table.loc["v10", "POOLED"] - table.loc[top, "POOLED"]
    wins = int((table.loc[top, blocks] < table.loc["v10", blocks]).sum())
    print(f"best={top} gain={gain:.4f} block_wins={wins}/3 -> {'PASS' if gain >= 0.005 and wins >= 2 else 'FAIL'}")


if __name__ == "__main__":
    main()
