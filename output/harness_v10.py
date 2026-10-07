"""Feature test on the release-episode candidate (D1 >=50% of peak cinemas x3d).

A: film-level D1-D3 features across all cinemas (test: all of test_history).
B: calendar features (D1 weekday, off days in D1-D3, holiday flag on target).
Variants are scored on the harness validation blocks; harness.py is unchanged.
"""

import numpy as np
import pandas as pd

import harness as h
import modelv5 as m5

FEATURES_A = [
    "film_log_ticket_d1", "film_log_ticket_d2", "film_log_ticket_d3",
    "film_cinemas_d1", "film_cinemas_d2", "film_cinemas_d3",
    "film_growth", "pair_film_share", "pair_scale_vs_film",
]
FEATURES_B = ["d1_dow", "offdays_d1_d3", "target_is_holiday"]
VARIANTS = {
    "base": h.FEATURES_CANDIDATE,
    "+A": h.FEATURES_CANDIDATE + FEATURES_A,
    "+B": h.FEATURES_CANDIDATE + FEATURES_B,
    "+A+B": h.FEATURES_CANDIDATE + FEATURES_A + FEATURES_B,
}


def film_early_stats(records, d1):
    """Per-film D1-D3 totals over every cinema the film plays in."""
    early = h.with_day_index(records[records["movie_title"].isin(d1.index)], d1)
    early = early[early["k"].between(1, 3)]
    daily = early.groupby(["movie_title", "k"]).agg(
        ticket=("total_ticket", "sum"), cinemas=("cinema_ids", "nunique")
    ).unstack("k").reindex(columns=pd.MultiIndex.from_product([["ticket", "cinemas"], [1, 2, 3]]), fill_value=0).fillna(0)
    stats = pd.DataFrame(index=daily.index)
    for day in (1, 2, 3):
        stats[f"film_log_ticket_d{day}"] = np.log1p(daily[("ticket", day)])
        stats[f"film_cinemas_d{day}"] = daily[("cinemas", day)]
    stats["film_growth"] = daily[("ticket", 3)] / daily[("ticket", 1)].clip(lower=1)
    stats["film_total"] = daily["ticket"].sum(axis=1)
    pair_scale = (early.groupby(h.PAIR_KEYS)["total_ticket"].sum() / 3).clip(lower=1)
    stats["film_median_scale"] = pair_scale.groupby(level="movie_title").median()
    return stats


def add_film_features(frame, stats):
    out = frame.merge(stats, left_on="movie_title", right_index=True, how="left")
    pair_total = out[["d1_ticket", "d2_ticket", "d3_ticket"]].sum(axis=1)
    out["pair_film_share"] = pair_total / out["film_total"].clip(lower=1)
    out["pair_scale_vs_film"] = out["scale"] / out["film_median_scale"]
    return out


def add_calendar_features(frame, holidays):
    out = frame.copy()
    d1 = pd.to_datetime(out["history_start"])
    calendar = holidays.drop_duplicates("date").set_index("date")
    off = calendar["day_tipe"].eq("weekend") | calendar["holiday_tipe"].eq("holiday")
    out["d1_dow"] = d1.dt.dayofweek
    out["offdays_d1_d3"] = sum(
        (d1 + pd.Timedelta(days=i)).map(off).fillna((d1 + pd.Timedelta(days=i)).dt.dayofweek >= 5).astype(int)
        for i in range(3)
    )
    out["target_is_holiday"] = out["target_date"].map(calendar["holiday_tipe"].eq("holiday")).fillna(False).astype(int)
    return out


def prepare_train(train, test, movies, holidays, prices):
    min_shows, frac, days = h.CHOSEN_D1_DEFINITION
    d1 = h.eligible_release_dates(h.wide_release_dates(train, min_shows, frac, days), train, set(test["movie_title"]))
    source = h.preview_free_source(train, d1)
    episodes = h.build_episodes(train, d1)
    episodes = m5.add_features(episodes, source, movies, holidays, prices)
    episodes = add_film_features(episodes, film_early_stats(source, d1))
    return add_calendar_features(episodes, holidays), source


def prepare_test(test, history, source, movies, holidays, prices):
    rows = m5.make_test_rows(test, history)
    rows = m5.add_features(rows, source, movies, holidays, prices)
    rows = add_film_features(rows, film_early_stats(history, history.groupby("movie_title")["date_show"].min()))
    return add_calendar_features(rows, holidays)


def clean(frames):
    h.encode_categories(frames)
    for f in frames:
        cols = FEATURES_A + FEATURES_B
        f[cols] = f[cols].replace([np.inf, -np.inf], np.nan)


def main():
    train, test, history, movies, holidays, prices, _ = m5.read_inputs()
    train = m5.aggregate_daily_transactions(train)
    episodes, _ = prepare_train(train, test, movies, holidays, prices)
    clean([episodes])

    errors = {name: {} for name in VARIANTS}
    for label, start, mask in h.block_masks(episodes):
        block = episodes[mask]
        fit = episodes[episodes["window_end"] < start]
        for name, features in VARIANTS.items():
            errors[name][label] = h.scaled_error(block["target"], h.fit_predict(fit, block, features), block["scale"])
    rows = []
    for name, per_block in errors.items():
        rows.append({"variant": name, **{k: v.mean() for k, v in per_block.items()},
                     "POOLED": np.concatenate(list(per_block.values())).mean()})
    table = pd.DataFrame(rows).set_index("variant")
    print(table.round(4).to_string())

    base = table.loc["base"]
    best = table.drop(index="base")["POOLED"].idxmin()
    wins = int((table.loc[best, [b[0] for b in h.VALIDATION_BLOCKS]] < base[[b[0] for b in h.VALIDATION_BLOCKS]]).sum())
    gain = base["POOLED"] - table.loc[best, "POOLED"]
    passed = gain >= 0.01 and wins >= 2
    print(f"best={best} gain={gain:.4f} block_wins={wins}/3 -> {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
