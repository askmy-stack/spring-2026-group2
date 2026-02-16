from pathlib import Path
import pandas as pd

p = Path("results/bids_dataset/sub-001/eeg/sub-001_task-extremeversustraditionalvideos_channels.tsv")
df = pd.read_csv(p, sep="\t")

# Ensure required columns exist
for col in ["type", "units"]:
    if col not in df.columns:
        df[col] = ""

df["type"] = "EEG"
df["units"] = "V"

df.to_csv(p, sep="\t", index=False)
print("Fixed:", p)
print(df.head())
