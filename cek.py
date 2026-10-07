import pandas as pd

scored = {"kaggle 0.58309": "output/kaggle_058309.csv", "kaggle 0.56079": "output/kaggle_056079.csv"}
local = {"v5": "output/submission_modelv5.csv", "v9": "output/submission_modelv9.csv"}

for sname, spath in scored.items():
    s = pd.read_csv(spath).sort_values("id").reset_index(drop=True)
    t = s["total_ticket"]
    print(f"\n{sname}: mean={t.mean():.2f} zero={(t == 0).mean():.3f} max={t.max():.1f}")
    for lname, lpath in local.items():
        l = pd.read_csv(lpath).sort_values("id").reset_index(drop=True)["total_ticket"]
        print(f"  vs {lname}: korelasi={t.corr(l):.4f} selisih_maks={(t - l).abs().max():.4f}")