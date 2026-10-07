"""Release-episode model v10: modelv9 plus film-level D1-D3 and calendar features.

Same episodes and model as modelv9 (D1 >=50% of peak cinemas for 3 days). Adds
harness_v10 feature sets A (film totals, cinema counts, growth, pair share and
scale vs film median; test uses all of test_history) and B (D1 weekday, off
days in D1-D3, holiday flag on the target date). Harness: pooled MASE 0.403 -> 0.369.
"""

import numpy as np
import pandas as pd

import harness as h
import harness_v10 as hv
import modelv5 as m5


def main():
    train, test, history, movies, holidays, prices, sample = m5.read_inputs()
    train = m5.aggregate_daily_transactions(train)
    history = m5.aggregate_daily_transactions(history)

    episodes, source = hv.prepare_train(train, test, movies, holidays, prices)
    print(f"TRAIN EPISODES: films={episodes['movie_title'].nunique()} pairs={len(episodes) // 7:,} rows={len(episodes):,}")
    test_rows = hv.prepare_test(test, history, source, movies, holidays, prices)
    print(f"TEST EPISODES: films={test_rows['movie_title'].nunique()} pairs={len(test_rows) // 7:,} rows={len(test_rows):,}")

    hv.clean([episodes, test_rows])
    pred = h.fit_predict(episodes, test_rows, hv.VARIANTS["+A+B"])
    submission = (
        pd.DataFrame({"id": test_rows["id"].to_numpy(), "total_ticket": np.maximum(pred, 0)})
        .set_index("id")
        .reindex(sample["id"])
        .reset_index()
    )
    if not submission["id"].equals(sample["id"]) or submission["total_ticket"].isna().any():
        raise ValueError("Submission IDs must match sample_submission.csv with no missing predictions")
    output_path = m5.OUTPUT_DIR / "submission_modelv10.csv"
    submission.to_csv(output_path, index=False)

    saved = pd.read_csv(output_path)
    print("\nSaved to:", output_path)
    print("Rows:", len(saved), "| NaN:", int(saved["total_ticket"].isna().sum()),
          "| negative:", int((saved["total_ticket"] < 0).sum()),
          f"| zero share: {saved['total_ticket'].eq(0).mean():.4f}",
          f"| max: {saved['total_ticket'].max():.2f}")
    v9_path = m5.OUTPUT_DIR / "submission_modelv9.csv"
    if v9_path.is_file():
        both = saved.merge(pd.read_csv(v9_path), on="id", suffixes=("_v10", "_v9"))
        print(f"Correlation with modelv9: pearson={both['total_ticket_v10'].corr(both['total_ticket_v9']):.4f}")


if __name__ == "__main__":
    main()
