"""
Plot the distribution of exclusion reasons across train, val, and test sets.
Values are normalized by the total number of excluded articles in each set.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "dataset/data_abstract_only"
SPLITS = {
    "train": f"{DATA_DIR}/train_data.csv",
    "val": f"{DATA_DIR}/val_data.csv",
    "test": f"{DATA_DIR}/test_data.csv",
}
REASON_COL = "If 0, reason to reject?"
COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}


def load_reason_counts(path: str) -> pd.Series:
    df = pd.read_csv(path)
    reasons = df[REASON_COL].dropna()
    # Some entries have multiple comma-separated reasons; split and explode them
    reasons = reasons.str.split(", ").explode().str.strip()
    return reasons.value_counts()


counts = {split: load_reason_counts(path) for split, path in SPLITS.items()}

# Union of all reasons, sorted by total frequency
all_reasons = sorted(
    set().union(*[c.index for c in counts.values()]),
    key=lambda r: -sum(c.get(r, 0) for c in counts.values()),
)

# Normalize by total excluded articles per split
normalized = {}
for split, c in counts.items():
    total = c.sum()
    normalized[split] = pd.Series(
        {r: c.get(r, 0) / total for r in all_reasons}
    )

# Bar plot
x = np.arange(len(all_reasons))
n = len(SPLITS)
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))

for i, (split, vals) in enumerate(normalized.items()):
    offset = (i - (n - 1) / 2) * width
    ax.bar(x + offset, vals.values, width, label=split, color=COLORS[split])

ax.set_xticks(x)
ax.set_xticklabels(all_reasons, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Proportion of excluded articles")
ax.set_title("Distribution of exclusion reasons (normalized)")
ax.legend(title="Split")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

plt.tight_layout()
plt.savefig("exclusion_reasons_distribution.png", dpi=150)
print("Saved to exclusion_reasons_distribution.png")
plt.show()

# --- Textual summary ---
MINORITY_THRESHOLD = 10  # flag categories with fewer than this many examples

print("\n" + "=" * 70)
print("EXCLUSION REASON SUMMARY")
print("=" * 70)

totals = {split: c.sum() for split, c in counts.items()}
for split, total in totals.items():
    print(f"  {split:6s}: {total} excluded articles")

print()
print(f"{'Reason':<55} {'train':>6} {'val':>5} {'test':>5}  flags")
print("-" * 80)

minority_flags = []
for reason in all_reasons:
    row_counts = {s: counts[s].get(reason, 0) for s in SPLITS}
    flags = []
    for split, n_count in row_counts.items():
        if n_count < MINORITY_THRESHOLD:
            flags.append(f"{split}:{n_count}")
    flag_str = "  *** MINORITY: " + ", ".join(flags) if flags else ""
    print(
        f"  {reason:<53} {row_counts['train']:>6} {row_counts['val']:>5} {row_counts['test']:>5}{flag_str}"
    )
    if flags:
        minority_flags.append((reason, row_counts))

print()
if minority_flags:
    print(f"Categories with fewer than {MINORITY_THRESHOLD} examples in at least one split:")
    for reason, row_counts in minority_flags:
        parts = [f"{s}={row_counts[s]}" for s in SPLITS]
        print(f"  - {reason} ({', '.join(parts)})")
else:
    print("No minority categories found.")
