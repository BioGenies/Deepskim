import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parents[2] / "dataset" / "data_preprocessed.csv"

df = pd.read_csv(DATA_PATH)

col = "If so; reason to reject?"

counts = df[col].value_counts(dropna=False)

multi = df[col].dropna().str.contains(",").sum()

print(f"Column: '{col}'")
print(f"Total rows: {len(df)}")
print(f"Entries with multiple exclusion criteria: {multi}\n")
print(f"{'Value':<50} {'Count':>7}")
print("-" * 59)
for value, count in counts.items():
    label = "(empty/NaN)" if pd.isna(value) else repr(value)
    print(f"{label:<50} {count:>7}")