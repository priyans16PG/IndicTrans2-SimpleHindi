# IndicTrans2-SimpleHindi

> **Version:** `0.1.3 (beta)`  
> **Status:** 🚧 *Beta — Active Development*

**IndicTrans2-SimpleHindi** is a lightweight LoRA fine-tuned adapter for `ai4bharat/indictrans2-en-indic-dist-200M` designed specifically for **everyday, colloquial spoken Hindi**.

Standard machine translation models often produce overly formal, archaic, or bookish translations (e.g., using *कृपया*, *प्रसन्नता*, *दूरभाष यंत्र*). **IndicTrans2-SimpleHindi** bridges this gap by shifting the output distribution to natural modern Hindi with everyday idioms and natural Devanagari loanwords (e.g., *टेंशन*, *प्लान*, *ऑफिस*, *लाइटिंग*, *मीटिंग*).

---

## Key Features

- **Colloquial Everyday Hindi:** Outputs sound like natural conversational speech rather than textbook literature.
- **Natural Loanword Spelling:** Common modern terms are naturally transcribed in Devanagari (*ऑर्डर*, *टेंशन*, *कंट्रोल*, *वॉट्सऐप*, *लोकेशन*).
- **Anti-Catastrophic Forgetting Mix:** Fine-tuned on a balanced multi-source blend (~13,500 pairs) containing conversational dialogues, Tatoeba colloquial sentences, and benchmark general pairs so the model retains formal and long-sentence translation competence.
- **Lightweight LoRA Architecture:** Modifies attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) with only **3.5M trainable parameters (~1.64%)**, requiring only ~14 MB for the adapter weights.
- **Ready for CPU & GPU:** Auto-selects CUDA when available (with FP16 precision) and falls back cleanly to CPU.

---

## Benchmark Comparisons

| English Input | Default IndicTrans2 (Formal) | IndicTrans2-SimpleHindi (v0.1.3 beta) |
|---|---|---|
| *Don't worry about it, I will handle it.* | इसके बारे में चिंता मत करो, मैं इसे संभालूंगा। | चिंता मत करो, मैं संभाल लूंगा। |
| *Where are you going right now?* | आप अभी कहाँ जा रहे हैं? | तुम अभी कहाँ जा रहे हो? |
| *Don't take tension, everything is under control.* | तनाव मत लो, सब कुछ नियंत्रण में है। | टेंशन मत लो, सब कुछ कंट्रोल में है। |
| *Can you send me the location on WhatsApp?* | क्या आप मुझे व्हाट्सएप पर स्थान भेज सकते हैं? | क्या तुम मुझे वो लोकेशन वॉट्सऐप पर भेज सकते हो? |

---

## Quickstart

### 1. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/priyans16PG/IndicTrans2-SimpleHindi.git
cd IndicTrans2-SimpleHindi
pip install -r requirements.txt
```

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

EN > How are you doing today?
HI > आज कैसा चल रहा है?

EN > Don't worry about it, I will handle it.
HI > चिंता मत करो, मैं संभाल लूंगा।

EN > Can you send me the location on WhatsApp?
HI > क्या तुम मुझे वो लोकेशन वॉट्सऐप पर भेज सकते हो?
```

---

### 3. Python API / Batch Inference

```python
from inference import translate_batch

sentences = [
    "Don't worry about it, I will handle it.",
    "Can you send me the location on WhatsApp?"
]

translations = translate_batch(sentences)
for src, tgt in zip(sentences, translations):
    print(f"{src} -> {tgt}")
```

---

## Dataset & Training

The included dataset is located in the [`data/`](./data) directory:

- **`train_data.csv`**: **13,061** training pairs combining conversational dialogues, Tatoeba everyday sentences, chunked dialogue splits, and benchmark general pairs.
- **`val_data.csv`**: **500** held-out evaluation pairs.
- **`daily_hindi_mixed.csv`**: Full **13,561** deduplicated pairs.
- **`daily_hindi_conversational.csv`**: Curated pure conversational source lines.

To re-train or fine-tune with custom parameters:

```bash
python train.py
```

### Training Configuration & Metrics
- **Base Model:** `ai4bharat/indictrans2-en-indic-dist-200M`
- **LoRA Parameters:** $r = 16, \alpha = 32, \text{dropout} = 0.05$
- **Total Dataset Size:** 13,561 pairs
- **Validation Loss Progress:**
  - Initial Phase (5.5k pairs): `2.509` $\rightarrow$ `2.278` $\rightarrow$ `2.207`
  - Extended Phase with Tatoeba (12.9k pairs): `1.944` $\rightarrow$ **`1.927`**
  - Continuation Phase with Dialogue Chunking (13.5k pairs): **`1.973`** *(robust colloquial idiom & dialogue coverage)*

---

## Project Structure

```
IndicTrans2-SimpleHindi/
├── model/                     # Pre-trained LoRA adapter weights (v0.1.3 beta)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── data/                      # Training and validation datasets (~13.5k pairs)
│   ├── train_data.csv
│   ├── val_data.csv
│   ├── daily_hindi_mixed.csv
│   └── daily_hindi_conversational.csv
├── chat.py                    # Interactive terminal translation shell
├── inference.py               # Batch inference script
├── train.py                   # LoRA fine-tuning training script
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## Release Notes

### v0.1.3 (beta)
- Integrated pre-processed and chunked dialogue pairs from newly added dataset, expanding total dataset to **13,561 pairs** (13,061 train / 500 val).
- Dialogue chunks cleaned of stage directions, annotations, and speaker labels with length constrained $\le 128$ tokens.
- Enhanced regex post-processing in `inference.py` and `chat.py` to automatically strip quotation/dialogue formatting artifacts.
- Continued LoRA fine-tuning for 2 epochs, maintaining stable eval loss while expanding colloquial coverage and everyday tone.

### v0.1.2 (beta)
- Merged **Tatoeba Hindi-English dataset** (6,072 sentences), expanding total dataset to **12,913 pairs**.
- Continued LoRA fine-tuning: validation loss decreased to a new best of **`1.927`** (down from `2.207`).
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
