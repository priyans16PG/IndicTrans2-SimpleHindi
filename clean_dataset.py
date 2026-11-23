import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Read from backup or original data files
train_path = BASE_DIR / "data" / "train_data.csv"
val_path = BASE_DIR / "data" / "val_data.csv"

# Re-read raw combined dataset if available or current
if (BASE_DIR.parent / "daily_hindi_combined_v3.csv").exists():
    raw_df = pd.read_csv(BASE_DIR.parent / "daily_hindi_combined_v3.csv")
    print(f"Re-loading from daily_hindi_combined_v3.csv: {len(raw_df)} rows")
else:
    raw_df = pd.read_csv(BASE_DIR / "data" / "daily_hindi_mixed.csv")

# Regex for true stage directions
stage_re_en = re.compile(
    r'^(\[|\()|'                               # Starts with bracket/paren
    r'^(act\s+[ivxlcdm\d]+|scene\s+[ivxlcdm\d]+|exeunt|exit)(\.|\:)?$|'  # Act I, Scene II, Exeunt
    r'^enter\s+[A-Z][a-zA-Z\s,]+(\.|\:)?$|'   # Enter Romeo, Mercutio
    r'^(a\s+street|a\s+public\s+place|another\s+room|an\s+open\s+place)\.?$',  # Stage settings
    re.IGNORECASE
)
stage_re_hi = re.compile(
    r'^(अधिनियम\s+[ivxlcdm\d]+|दृश्य\s+[ivxlcdm\d]+)|'
    r'(प्रवेश\s+करते\s+हैं|प्रस्थान\s+करते\s+हैं|प्रवेश\s+करती\s+हैं|प्रस्थान\s+करती\s+हैं)',
    re.IGNORECASE
)

def evaluate_row(en, hi):
    en_s = str(en).strip()
    hi_s = str(hi).strip()

    if not en_s or not hi_s or en_s.lower() == "nan" or hi_s.lower() == "nan":
        return False, "empty_or_nan"

    # Stage directions
    if stage_re_en.search(en_s):
        return False, "stage_direction_en"
    if stage_re_hi.search(hi_s) and len(en_s) < 25:
        return False, "stage_direction_hi"

    en_l = len(en_s)
    hi_l = len(hi_s)
    ratio = hi_l / max(en_l, 1)

    # Extreme length mismatches (e.g. Shakespeare paragraph paired with one short sentence, or vice versa)
    if en_l > 40 and ratio < 0.25 and (en_l - hi_l) > 30:
        return False, f"extreme_short_hi (en:{en_l}, hi:{hi_l})"
    if en_l < 35 and hi_l > 100 and ratio > 3.5:
        return False, f"extreme_long_hi (en:{en_l}, hi:{hi_l})"

    return True, "ok"

valid_mask = []
reasons = []
for _, row in raw_df.iterrows():
    ok, reason = evaluate_row(row["en"], row["hi"])
    valid_mask.append(ok)
    reasons.append(reason)

raw_df["reason"] = reasons
removed = raw_df[~pd.Series(valid_mask, index=raw_df.index)]

print(f"\nTotal removed corrupted/stage rows: {len(removed)}")
for _, r in removed.head(20).iterrows():
    print(f"   [{r['reason']}] EN: {r['en']} | HI: {r['hi']}")

clean_df = raw_df[pd.Series(valid_mask, index=raw_df.index)].copy()
clean_df["en"] = clean_df["en"].astype(str).str.strip()
clean_df["hi"] = clean_df["hi"].astype(str).str.strip()

# Deduplicate
clean_df = clean_df.drop_duplicates(subset=["en", "hi"])
clean_df = clean_df.drop_duplicates(subset=["en"], keep="first")
clean_df = clean_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

print(f"\nTotal clean deduplicated rows: {len(clean_df)}")

# Create 500 val split and remainder train split
val_df = clean_df.iloc[:500][["en", "hi"]]
train_df = clean_df.iloc[500:][["en", "hi"]]

print(f"Train rows: {len(train_df)}")
print(f"Val rows: {len(val_df)}")

# Save to IndicTrans2-SimpleHindi/data/
train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
clean_df[["en", "hi"]].to_csv(BASE_DIR / "data" / "daily_hindi_mixed.csv", index=False)

# Sync to root directory
root_dir = BASE_DIR.parent
train_df.to_csv(root_dir / "train_data.csv", index=False)
val_df.to_csv(root_dir / "val_data.csv", index=False)
print("Saved cleanly to data directories.")

