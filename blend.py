import pandas as pd

v5 = pd.read_csv("output/submission_modelv5.csv")
v9 = pd.read_csv("output/submission_modelv9.csv")
assert v5["id"].equals(v9["id"])

blend = v9.copy()
blend["total_ticket"] = 0.5 * v5["total_ticket"] + 0.5 * v9["total_ticket"]
blend.to_csv("output/submission_blend50.csv", index=False)

zero = v9.copy()
zero["total_ticket"] = 0.0
zero.to_csv("output/submission_zero.csv", index=False)

print("selesai:", blend.shape, zero.shape)