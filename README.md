# Cinema Ticket Demand Forecast

Forecast daily ticket sales for each movie and cinema pair. The competition task uses the first three observed days (D1–D3) to predict the following seven days (D4–D10).

## Metric

The score is Mean Absolute Scaled Error (MASE). For each movie/cinema pair, the scale is `max(mean(D1, D2, D3), 1)`. The final score is the mean of each forecast row's absolute error divided by that pair's scale. A lower MASE is better; lower predicted ticket totals do not automatically mean a better model.

## Files

- `output/modelv2.py`: prior model and feature experiments.
- `output/modelv3.py`: reproducible temporal validation, model selection, final training, and submission generation.
- `output/modelv4.py`: adds as-of pair/cinema demand history, pair/cinema/movie categories, and horizon calibration selected with walk-forward MASE.
- `output/modelv5.py`: aligns training windows with missing calendar days in test history, uses observed shows/occupancy and holiday/price features, and compares against modelv4 with temporal MASE validation.
- `input/`: local competition data; CSVs are intentionally excluded from Git.
- `output/`: generated submission files; CSVs are intentionally excluded from Git.

## Setup

Use Python 3.11 or newer, install the requirements, and place the competition CSVs in `input/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The required input files are `train.csv`, `test.csv`, `test_history.csv`, `movies.csv`, `holidays.csv`, `ticket_prices.csv`, and `sample_submission.csv`.

## Run modelv5

From the repository root:

```powershell
python output/modelv5.py
```

The script uses `SEED = 2026`, aggregates transaction data to one movie/cinema/day row, compares modelv4's complete-record training windows with modelv5's zero-filled calendar windows on four chronological validation origins, and selects by walk-forward MASE. Missing D1-D3 transaction days are treated as `0` tickets, with missing-day indicators kept as features. Modelv5 also uses first-three-day show/occupancy, holiday, and ticket-price features. It writes `output/submission_modelv5.csv` and prints submission and final sanity checks.

The local validation MASE helps compare model variants, but it is not the Kaggle leaderboard score and does not guarantee public leaderboard performance.
