import pandas as pd

FALSE_PREDS = "strong_false_predictions.csv"
OUTPUT = "false_predictions_analysis.csv"
TEST_DATA = "src/test_data.csv"
METADATA = "dataset/amyloid_sheet_16Oct_metadata.csv"

false_preds = pd.read_csv(FALSE_PREDS)
test_data = pd.read_csv(TEST_DATA)
metadata = pd.read_csv(METADATA, usecols=["PMID", "Decided by what?"])

# Normalize titles for matching
test_data["Title_norm"] = test_data["Title"].str.strip().str.lower()
false_preds["Title_norm"] = false_preds["Title"].str.strip().str.lower()

title_to_pmid = test_data.set_index("Title_norm")["PMID"].to_dict()
pmid_to_decided = metadata.dropna(subset=["PMID"]).set_index("PMID")["Decided by what?"].to_dict()

print(f"{'FP/FN':<5}  {'PMID':<12}  {'Decided by what?':<40}  Title")
print("-" * 120)

rows = []
unmatched_titles = []

for _, row in false_preds.iterrows():
    title = row["Title"]
    fp_fn = row["FP/FN"]
    pmid = title_to_pmid.get(row["Title_norm"])

    if pmid is None:
        unmatched_titles.append(title)
        continue

    decided = pmid_to_decided.get(pmid)
    if pd.isna(decided) if decided is not None else True:
        decided = None

    print(f"{fp_fn:<5}  {str(pmid):<12}  {str(decided or '—'):<40}  {title[:80]}")
    rows.append({"FP/FN": fp_fn, "PMID": pmid, "Title": title, "Decided by what?": decided})

pd.DataFrame(rows).to_csv(OUTPUT, index=False)
print(f"\nSaved to {OUTPUT}")

if unmatched_titles:
    print(f"\n[WARNING] {len(unmatched_titles)} title(s) not found in test_data.csv:")
    for t in unmatched_titles:
        print(f"  - {t[:100]}")
