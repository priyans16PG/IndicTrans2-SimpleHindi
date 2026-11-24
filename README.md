# IndicTrans2-SimpleHindi

> **Version:** `0.1.3 (beta)`  
> **Status:** 🚀 *Beta — Stable & Verified (30/30 Benchmark Passed)*

**IndicTrans2-SimpleHindi** is a lightweight LoRA fine-tuned adapter for `ai4bharat/indictrans2-en-indic-dist-200M` designed specifically for **everyday, colloquial spoken Hindi**.

Standard machine translation models often produce overly formal, archaic, or bookish translations (e.g., using *कृपया*, *प्रसन्नता*, *समय सीमा*, *स्थान*). **IndicTrans2-SimpleHindi** bridges this gap by shifting the output distribution to natural modern Hindi with everyday idioms and natural Devanagari loanwords (e.g., *टेंशन*, *प्लान*, *ऑफिस*, *मीटिंग*, *डेडलाइन*, *वॉट्सऐप*, *लोकेशन*, *कॉल करना*).

---

## Key Features

- **Colloquial Everyday Hindi:** Outputs sound like natural conversational speech rather than textbook literature.
- **Natural Loanword Spelling:** Common modern terms are naturally transcribed in Devanagari (*ऑर्डर*, *टेंशन*, *कंट्रोल*, *वॉट्सऐप*, *लोकेशन*, *डेडलाइन*, *मीटिंग*).
- **100% Benchmark Coherence:** Scored **30/30 (100%)** on the comprehensive multi-clause test suite with zero repetition loops and zero cause-and-effect inversion.
- **Foreign Script Token Suppression:** Built-in dynamic vocabulary filtering suppresses all 19k+ non-Devanagari / Arabic / Sindhi tokens in IndicTrans2's multilingual vocabulary during generation.
- **Lightweight LoRA Architecture:** Modifies attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) with only **1.77M trainable parameters (~0.83%)**, requiring only ~7 MB for the adapter weights.
- **Validation Loss:** Reached a new all-time low of **`1.312`** (down from `1.927` in v0.1.2 and `2.207` in v0.1.1).
- **Ready for CPU & GPU:** Auto-selects CUDA when available (with FP16 precision) and falls back cleanly to CPU.

---

## Benchmark Comparisons

| English Input | Default IndicTrans2 (Formal) | IndicTrans2-SimpleHindi (v0.1.3 beta) |
|---|---|---|
| *Don't worry about it, I will handle everything.* | इसके बारे में चिंता मत करो, मैं सब कुछ संभालूंगा। | चिंता मत करो, मैं सब संभाल लूंगा। |
| *Where are you right now? We have been waiting for twenty minutes.* | आप अभी कहाँ हैं? हम बीस मिनट से इंतज़ार कर रहे हैं। | तुम अभी कहाँ हो? हम बीस मिनट से इंतज़ार कर रहे हैं। |
| *I am stuck in traffic right now, so I will be ten minutes late for the meeting.* | मैं अभी यातायात में फंसा हुआ हूं, इसलिए मुझे बैठक के लिए दस मिनट की देरी होगी। | मैं अभी ट्रैफिक में फंसा हुआ हूँ, इसलिए मीटिंग के लिए दस मिनट लेट हो जाऊँगा। |
| *If you don't start working on the project today, you won't finish before the deadline.* | यदि आप आज से प्रोजेक्ट पर काम शुरू नहीं करेंगे, तो आप समय सीमा से पहले काम पूरा नहीं करेंगे। | अगर तुम आज प्रोजेक्ट पर काम शुरू नहीं करोगे, तो तुम डेडलाइन से पहले खत्म नहीं करोगे। |
| *Can you send me the location on WhatsApp?* | क्या आप मुझे व्हाट्सएप पर स्थान भेज सकते हैं? | क्या तुम मुझे वो लोकेशन वॉट्सऐप पर भेज सकते हो? |
| *Don't take tension about the interview; just stay calm and answer honestly.* | साक्षात्कार के बारे में तनाव मत लो; बस शांत रहो और ईमानदारी से जवाब दो। | इंटरव्यू को लेकर टेंशन मत लीजिए; बस शांत रहें और ईमानदारी से जवाब दें। |

---

## Quickstart

### 1. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/priyans16PG/IndicTrans2-SimpleHindi.git
cd IndicTrans2-SimpleHindi
pip install -r requirements.txt
```

> [!IMPORTANT]
> **Transformers & Library Requirements:** You **must** use `transformers==4.46.3`, `peft==0.13.2`, `datasets==3.6.0`, and `huggingface-hub==0.36.2` as pinned in `requirements.txt`. Do **not** upgrade to Transformers 5.x; Transformers 5.x introduced breaking changes to attention mask utilities and cache formats that break IndicTrans2's custom model code, causing scrambled "bag-of-words" translations.

*(Note: Ensure you have accepted the model terms on Hugging Face for `ai4bharat/indictrans2-en-indic-dist-200M` and logged in via `huggingface-cli login`.)*

---

### 2. Interactive Chat Translator

To start an interactive command-line session:

```bash
python chat.py
```

```text
============================================================
 IndicTrans2-SimpleHindi v0.1.3 (beta)
 Running on: CUDA
============================================================

Model ready! Enter English text to translate to everyday conversational Hindi.
Type 'q' or 'exit' to quit.

EN > I am stuck in traffic right now, so I will be ten minutes late for the meeting.
HI > मैं अभी ट्रैफिक में फंसा हुआ हूँ, इसलिए मीटिंग के लिए दस मिनट लेट हो जाऊँगा।

EN > If you don't start working on the project today, you won't finish before the deadline.
HI > अगर तुम आज प्रोजेक्ट पर काम शुरू नहीं करोगे, तो तुम डेडलाइन से पहले खत्म नहीं करोगे।

EN > Can you send me the location on WhatsApp?
HI > क्या तुम मुझे वो लोकेशन वॉट्सऐप पर भेज सकते हो?
```

---

### 3. Python API / Batch Inference

```python
from inference import translate_batch

sentences = [
    "Don't worry about it, I will handle everything.",
    "I am stuck in traffic right now, so I will be ten minutes late for the meeting.",
    "If you don't start working on the project today, you won't finish before the deadline."
]

translations = translate_batch(sentences)
for src, tgt in zip(sentences, translations):
    print(f"{src} -> {tgt}")
```

---

### 4. Running the 30-Sentence Benchmark

To evaluate all 30 sentences from the standard simple Hindi benchmark test suite:

```bash
python run_30_benchmark.py
```

---

## Dataset & Training

The training datasets are located in the [`data/`](./data) directory:

- **`train_data.csv`**: **1,940** continuation training pairs combining book dialogue splits, curated conversational lines, and targeted modern loanword pairs.
- **`val_data.csv`**: **200** held-out evaluation pairs.
- **`daily_hindi_mixed.csv`**: Deduplicated archive of past conversational pairs.
- **`daily_hindi_conversational.csv`**: Pure conversational source lines.

To re-train or fine-tune with custom parameters:

```bash
python train.py
```

### Training Configuration & Metrics
- **Base Model:** `ai4bharat/indictrans2-en-indic-dist-200M`
- **Starting Weights:** Continuation fine-tuning from v0.1.2 adapter (`is_trainable=True`)
- **LoRA Parameters:** $r = 8, \alpha = 16, \text{dropout} = 0.05$ (1.77M trainable parameters, ~0.83% of total)
- **Target Preprocessing:** `is_target=True` (pure normalized Hindi token stream without leaking language codes)
- **Inference Anti-Repetition:** `no_repeat_ngram_size=3`, `repetition_penalty=1.2`
- **Target Vocabulary Suppression:** Non-Devanagari token suppression via `suppress_tokens`
- **Validation Loss:** **`1.312`** (2 epochs cosine schedule)
- **Benchmark Score:** **30 / 30 Passed (100%)**

---

## Project Structure

```
IndicTrans2-SimpleHindi/
├── model/                     # Fine-tuned LoRA adapter weights (v0.1.3 beta)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── data/                      # Cleaned training and validation datasets
│   ├── train_data.csv
│   ├── val_data.csv
│   ├── daily_hindi_mixed.csv
│   └── daily_hindi_conversational.csv
├── chat.py                    # Interactive terminal translation shell
├── inference.py               # Batch inference script
├── train.py                   # LoRA fine-tuning training script
├── run_30_benchmark.py        # 30-sentence benchmark evaluation script
├── debug_compare.py           # Side-by-side diagnostic script (Base vs. LoRA)
├── build_v013_data.py         # Dataset curation & filtering script
├── requirements.txt           # Pinned Python dependencies
├── .gitignore
└── README.md
```

---

## Release Notes

### v0.1.3 (beta)
- **Resolved Transformers 5.x Attention Incompatibility:** Pinned `transformers==4.46.3` and `peft==0.13.2` after isolating an upstream breaking change in Transformers 5.x attention mask / cache logic that scrambled encoder-decoder translations.
- **Targeted Simple Hindi Continuation Training:** Fine-tuned on a curated 2,140-pair continuation dataset blending cleaned book dialogues, conversational pairs, and targeted modern loanwords.
- **Vocabulary Modernization:** Shifted textbook words to natural colloquial speech (*यदि* $\rightarrow$ *अगर*, *समय सीमा* $\rightarrow$ *डेडलाइन*, *यातायात* $\rightarrow$ *ट्रैफिक*, *बैठक* $\rightarrow$ *मीटिंग*, *स्थान* $\rightarrow$ *लोकेशन*).
- **Multilingual Token Suppression:** Added automatic suppression of 19k+ non-Devanagari / Sindhi tokens in IndicTrans2's shared vocabulary, preventing cross-script leakage.
- **Record Evaluation Loss:** Validation loss improved to **`1.312`** (down from `1.927` in v0.1.2).
- **100% Benchmark Score:** Scored **30/30 (100%)** on the comprehensive simple Hindi test suite.
- **Clean Output Post-Processing:** Enhanced prefix stripping regex to eliminate residual language tags (`_Deva`) or numerical artifacts.

### v0.1.2 (beta)
- Merged **Tatoeba Hindi-English dataset** (6,072 sentences), expanding total dataset to **12,913 pairs**.
- Continued LoRA fine-tuning: validation loss decreased to **`1.927`** (down from `2.207`).
- Updated adapter weights with expanded vocabulary and better colloquial sentence coverage.

### v0.1.1 (beta)
- Initial public beta release with 5,500 conversational and general mix pairs.
- Pre-trained LoRA weights on 5,500 curated conversational and general mix pairs.
- Interactive `chat.py` CLI and `inference.py` helper.
- Full dataset and reproducibility scripts included.

---

## License & Attribution

- Built on top of [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) developed by AI4Bharat.
- Released under the MIT License.
