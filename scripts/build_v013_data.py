import re
import sys
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("=" * 80)
print("BUILDING v0.1.3 DATASET: CLEAN BOOK PAIRS + 25% CONVERSATIONAL DATA")
print("=" * 80)

# 1. Clean book data from dataset3_cleaned.csv
book_file = REPO_ROOT.parent / "dataset3_cleaned.csv"
if not book_file.exists():
    book_file = REPO_ROOT / "dataset3_cleaned.csv"

df_book = pd.read_csv(book_file)
stage_re_en = re.compile(
    r'^(\[|\()|'
    r'^(act\s+[ivxlcdm\d]+|scene\s+[ivxlcdm\d]+|exeunt|exit)(\.|\:)?$|'
    r'^enter\s+[A-Z]|'
    r'^(a\s+street|a\s+public\s+place|another\s+room|an\s+open\s+place)\.?$',
    re.I
)
stage_re_hi = re.compile(
    r'^(अधिनियम\s+[ivxlcdm\d]+|दृश्य\s+[ivxlcdm\d]+)|'
    r'(प्रवेश\s+करते|प्रस्थान\s+करते)',
    re.I
)

clean_book = []
for _, row in df_book.iterrows():
    en = str(row["en"]).strip()
    hi = str(row["hi"]).strip()
    if not en or not hi or len(en) < 5 or len(hi) < 5:
        continue
    if stage_re_en.search(en) or stage_re_hi.search(hi):
        continue
    ratio = len(hi) / max(len(en), 1)
    if len(en) > 40 and ratio < 0.25:
        continue
    if len(en) < 35 and len(hi) > 100 and ratio > 3.5:
        continue
    clean_book.append({"en": en, "hi": hi})

df_book_clean = pd.DataFrame(clean_book).drop_duplicates(subset=["en"])
print(f"1. Clean Book Dialogue Pairs: {len(df_book_clean)}")

# 2. Conversational sample (25% of daily_hindi_conversational.csv)
conv_file = REPO_ROOT / "data" / "daily_hindi_conversational.csv"
if not conv_file.exists():
    conv_file = SCRIPT_DIR / "data" / "daily_hindi_conversational.csv"
df_conv = pd.read_csv(conv_file)
df_conv_sample = df_conv.sample(n=1500, random_state=42)
print(f"2. Conversational Sample: {len(df_conv_sample)}")

# 3. Modern Simple Hindi targeted pairs for loanwords & colloquial terms
modern_pairs = [
    {"en": "If you don't start working on the project today, you won't finish before the deadline.", "hi": "अगर आप आज से प्रोजेक्ट पर काम शुरू नहीं करेंगे, तो आप डेडलाइन से पहले काम पूरा नहीं करेंगे।"},
    {"en": "When you remove distractions from your desk, it becomes much easier to focus on deep work.", "hi": "जब आप अपनी डेस्क से ध्यान भटकाने वाली चीज़ें हटा देते हैं, तो काम पर फोकस करना बहुत आसान हो जाता है।"},
    {"en": "Small habits may not seem important today, but over a year they make a huge difference.", "hi": "हो सकता है कि छोटी-छोटी आदतें आज ज़रूरी न लगें, लेकिन एक साल में वे बहुत बड़ा फर्क डालती हैं।"},
    {"en": "If you need any help, just let me know.", "hi": "अगर तुम्हें कोई मदद चाहिए, तो बस मुझे बता देना।"},
    {"en": "If it rains tomorrow, we will cancel the plan.", "hi": "अगर कल बारिश हुई, तो हम प्लान कैंसिल कर देंगे।"},
    {"en": "We have to submit the project before the deadline.", "hi": "हमें डेडलाइन से पहले प्रोजेक्ट सबमिट करना होगा।"},
    {"en": "The deadline is tomorrow at five PM.", "hi": "डेडलाइन कल शाम पांच बजे है।"},
    {"en": "You should focus on your studies right now.", "hi": "तुम्हें अभी अपनी पढ़ाई पर फोकस करना चाहिए।"},
    {"en": "It is very hard to focus with all this noise.", "hi": "इतने शोर में फोकस करना बहुत मुश्किल है।"},
    {"en": "I will call you after the meeting.", "hi": "मैं मीटिंग के बाद तुम्हें कॉल करूँगा।"},
    {"en": "Are you coming to the office today?", "hi": "क्या तुम आज ऑफिस आ रहे हो?"},
    {"en": "Send me the location on WhatsApp.", "hi": "मुझे वॉट्सऐप पर लोकेशन भेज दो।"},
    {"en": "Don't take tension, everything will be fine.", "hi": "टेंशन मत लो, सब ठीक हो जाएगा।"},
    {"en": "Don't waste your time worrying about things you cannot control.", "hi": "उन चीज़ों के बारे में चिंता करने में अपना टाइम बर्बाद मत करो जिन्हें आप कंट्रोल नहीं कर सकते।"},
    {"en": "My phone battery died, so I couldn't check your message on WhatsApp.", "hi": "मेरे फोन की बैटरी खत्म हो गई, इसलिए मैं आपका मैसेज वॉट्सऐप पर चेक नहीं कर सका।"},
    {"en": "Where are you right now? We have been waiting for twenty minutes.", "hi": "तुम अभी कहाँ हो? हम बीस मिनट से इंतज़ार कर रहे हैं।"},
    {"en": "I am stuck in traffic right now, so I will be ten minutes late for the meeting.", "hi": "मैं अभी ट्रैफिक में फंसा हुआ हूँ, इसलिए मीटिंग के लिए दस मिनट लेट हो जाऊँगा।"},
]
df_modern = pd.DataFrame(modern_pairs)
print(f"3. Targeted Simple Hindi Pairs: {len(df_modern)}")

# Merge & shuffle
combined = pd.concat([df_book_clean, df_conv_sample, df_modern]).drop_duplicates(subset=["en"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"\nTotal Clean v0.1.3 Dataset: {len(combined)} rows")

# Split: 200 for val, rest for train
val_size = 200
val_df = combined.iloc[:val_size][["en", "hi"]]
train_df = combined.iloc[val_size:][["en", "hi"]]

train_path = REPO_ROOT / "data" / "train_data.csv"
val_path = REPO_ROOT / "data" / "val_data.csv"
mixed_path = REPO_ROOT / "data" / "daily_hindi_mixed.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
combined[["en", "hi"]].to_csv(mixed_path, index=False)

print(f"Train split saved to {train_path} ({len(train_df)} rows)")
print(f"Val split saved to {val_path} ({len(val_df)} rows)")
print(f"Mixed dataset saved to {mixed_path} ({len(combined)} rows)")

# Sync to root
train_df.to_csv(REPO_ROOT.parent / "train_data.csv", index=False)
val_df.to_csv(REPO_ROOT.parent / "val_data.csv", index=False)
print("Synced to root train_data.csv and val_data.csv")
