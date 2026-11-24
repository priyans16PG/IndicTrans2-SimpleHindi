import re
import sys
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_PATH = REPO_ROOT / "data" / "train_data.csv"
print(f"Auditing: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
total_rows = len(df)
print(f"Total rows in train_data.csv: {total_rows}")

# 1. Empty or NaN rows
nan_mask = df["en"].isna() | df["hi"].isna() | (df["en"].astype(str).str.strip() == "") | (df["hi"].astype(str).str.strip() == "")
nan_rows = df[nan_mask]
print(f"\n1. Empty / NaN rows: {len(nan_rows)}")
if len(nan_rows) > 0:
    print(nan_rows.head(5))

# Clean df for further checks
clean_df = df[~nan_mask].copy()
clean_df["en"] = clean_df["en"].astype(str).str.strip()
clean_df["hi"] = clean_df["hi"].astype(str).str.strip()

# 2. Hindi column containing >20% Latin/English characters
def latin_char_ratio(text):
    if not text:
        return 0.0
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    # Count non-whitespace chars
    non_space_chars = len(re.findall(r'\S', text))
    return latin_chars / max(non_space_chars, 1)

clean_df["hi_latin_ratio"] = clean_df["hi"].apply(latin_char_ratio)
high_latin_rows = clean_df[clean_df["hi_latin_ratio"] > 0.20]
print(f"\n2. Hindi column with >20% Latin characters: {len(high_latin_rows)}")
if len(high_latin_rows) > 0:
    for _, row in high_latin_rows.head(10).iterrows():
        print(f"   [Ratio: {row['hi_latin_ratio']:.2f}] EN: {row['en']} | HI: {row['hi']}")

# 3. Severe length-ratio mismatches (Hindi > 2.5x longer or < 0.4x shorter than English by word count / char count)
# Using character count or word count:
clean_df["en_len"] = clean_df["en"].str.len()
clean_df["hi_len"] = clean_df["hi"].str.len()
clean_df["len_ratio"] = clean_df["hi_len"] / clean_df["en_len"].clip(lower=1)

mismatch_mask = (clean_df["len_ratio"] > 2.5) | (clean_df["len_ratio"] < 0.4)
# Filter out very short single-word phrases where char ratios naturally fluctuate (e.g., 'OK' vs 'ठीक है')
severe_mismatches = clean_df[mismatch_mask & (clean_df["en_len"] > 15)]
print(f"\n3. Severe length-ratio mismatches (>2.5x or <0.4x with len>15): {len(severe_mismatches)}")
if len(severe_mismatches) > 0:
    for _, row in severe_mismatches.head(10).iterrows():
        print(f"   [Ratio: {row['len_ratio']:.2f}] EN: {row['en']} | HI: {row['hi']}")

# 4. Duplicate English inputs with conflicting Hindi translations
dup_en = clean_df[clean_df.duplicated(subset=["en"], keep=False)].sort_values(by="en")
unique_translations_per_en = clean_df.groupby("en")["hi"].nunique()
conflicting_en = unique_translations_per_en[unique_translations_per_en > 1]
conflicting_rows = clean_df[clean_df["en"].isin(conflicting_en.index)]
print(f"\n4. Duplicate English inputs with conflicting Hindi translations: {len(conflicting_en)} distinct EN sentences ({len(conflicting_rows)} total rows)")
if len(conflicting_en) > 0:
    for en_text in list(conflicting_en.index)[:5]:
        variants = clean_df[clean_df["en"] == en_text]["hi"].tolist()
        print(f"   EN: {en_text}")
        for v in variants:
            print(f"       -> HI: {v}")

print("\n" + "="*60)
