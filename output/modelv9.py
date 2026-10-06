"""Release-episode model: train on wide-release-anchored episodes, predict test.

Train episodes follow the test construction verified in harness.py: D1 is the
film's wide-release date (>=50% of peak cinemas for 3 days), pairs need a
record on D3, D1-D3 and D4-D10 are zero-filled, and pre-D1 previews are never
used. Features come from the same functions as in the harness; cinema history
is computed from train.csv only, for train and test episodes alike.
"""

import numpy as np
import pandas as pd

import harness as h
import modelv5 as m5


def main():
    train, test, history, movies, holidays, prices, sample = m5.read_inputs()
    train = m5.aggregate_daily_transactions(train)
    history = m5.aggregate_daily_transactions(history)

    min_shows, frac, days = h.CHOSEN_D1_DEFINITION
    d1 = h.eligible_release_dates(
        h.wide_release_dates(train, min_shows, frac, days), train, set(test["movie_title"])
    )
    source = h.preview_free_source(train, d1)
    episodes = h.build_episodes(train, d1)
    episodes = m5.add_features(episodes, source, movies, holidays, prices)
    print(f"TRAIN EPISODES: films={episodes['movie_title'].nunique()} pairs={len(episodes) // 7:,} rows={len(episodes):,}")

    # D1 = first test date - 3 = film's first test_history date (verified for 100% of pairs).
    test_rows = m5.make_test_rows(test, history)
    test_rows = m5.add_features(test_rows, source, movies, holidays, prices)
    print(f"TEST EPISODES: films={test_rows['movie_title'].nunique()} pairs={len(test_rows) // 7:,} rows={len(test_rows):,}")

    h.encode_categories([episodes, test_rows])
    pred = h.fit_predict(episodes, test_rows, h.FEATURES_CANDIDATE)
    submission = (
        pd.DataFrame({"id": test_rows["id"].to_numpy(), "total_ticket": np.maximum(pred, 0)})
        .set_index("id")
        .reindex(sample["id"])
        .reset_index()
    )
    if not submission["id"].equals(sample["id"]) or submission["total_ticket"].isna().any():
        raise ValueError("Submission IDs must match sample_submission.csv with no missing predictions")
    output_path = m5.OUTPUT_DIR / "submission_modelv9.csv"
    submission.to_csv(output_path, index=False)

    saved = pd.read_csv(output_path)
    checks = saved.merge(test_rows[["id", "horizon", "scale"]], on="id")
    print("\nSaved to:", output_path)
    print("Rows:", len(saved), "| NaN:", int(saved["total_ticket"].isna().sum()),
          "| negative:", int((saved["total_ticket"] < 0).sum()),
          f"| zero share: {saved['total_ticket'].eq(0).mean():.4f}",
          f"| max: {saved['total_ticket'].max():.2f}")
    ratio = (checks["total_ticket"] / checks["scale"]).groupby(checks["horizon"]).mean()
    print("Mean pred/scale D4-D10:", [round(float(v), 3) for v in ratio])
    v5_path = m5.OUTPUT_DIR / "submission_modelv5.csv"
    if v5_path.is_file():
        both = saved.merge(pd.read_csv(v5_path), on="id", suffixes=("_v9", "_v5"))
        print(f"Correlation with modelv5: pearson={both['total_ticket_v9'].corr(both['total_ticket_v5']):.4f} "
              f"spearman={both['total_ticket_v9'].corr(both['total_ticket_v5'], method='spearman'):.4f}")


if __name__ == "__main__":
    main()
