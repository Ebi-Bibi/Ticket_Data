"""Release-episode model v11: modelv10 plus the release schedule of other films (G).

Same episodes, features and parameters as modelv10, plus counts of other films
whose D1 falls after this film's D3 (up to the target date, on the target date,
and in D4-D10), overall and for films sharing a genre. Only D1 dates and
movies.csv genres of other films are used. Harness: pooled MASE 0.3687 -> 0.3609.
"""

import numpy as np
import pandas as pd

import harness as h
import harness_v11 as hx
import modelv5 as m5


def main():
    train, test, history, movies, holidays, prices, sample = m5.read_inputs()
    train = m5.aggregate_daily_transactions(train)
    history = m5.aggregate_daily_transactions(history)

    episodes, test_rows = hx.prepare_g(train, test, history, movies, holidays, prices, with_test=True)
    print(f"TRAIN EPISODES: films={episodes['movie_title'].nunique()} pairs={len(episodes) // 7:,} rows={len(episodes):,}")
    print(f"TEST EPISODES: films={test_rows['movie_title'].nunique()} pairs={len(test_rows) // 7:,} rows={len(test_rows):,}")

    hx.hv.clean([episodes, test_rows])
    pred = h.fit_predict(episodes, test_rows, hx.G_VARIANTS["+G"])
    submission = (
        pd.DataFrame({"id": test_rows["id"].to_numpy(), "total_ticket": np.maximum(pred, 0)})
        .set_index("id")
        .reindex(sample["id"])
        .reset_index()
    )
    if not submission["id"].equals(sample["id"]) or submission["total_ticket"].isna().any():
        raise ValueError("Submission IDs must match sample_submission.csv with no missing predictions")
    output_path = m5.OUTPUT_DIR / "submission_modelv11.csv"
    submission.to_csv(output_path, index=False)

    saved = pd.read_csv(output_path)
    print("\nSaved to:", output_path)
    print("Rows:", len(saved), "| NaN:", int(saved["total_ticket"].isna().sum()),
          "| negative:", int((saved["total_ticket"] < 0).sum()),
          f"| zero share: {saved['total_ticket'].eq(0).mean():.4f}",
          f"| max: {saved['total_ticket'].max():.2f}")
    v10_path = m5.OUTPUT_DIR / "submission_modelv10.csv"
    if v10_path.is_file():
        both = saved.merge(pd.read_csv(v10_path), on="id", suffixes=("_v11", "_v10"))
        print(f"Correlation with modelv10: pearson={both['total_ticket_v11'].corr(both['total_ticket_v10']):.4f}")


if __name__ == "__main__":
    main()
