# IndicTrans2-SimpleHindi

> **Version:** `0.1.1 (beta)`  
> **Status:** 🚧 *Beta — Active Development*

**IndicTrans2-SimpleHindi** is a lightweight LoRA fine-tuned adapter for `ai4bharat/indictrans2-en-indic-dist-200M` designed specifically for **everyday, colloquial spoken Hindi**.

Standard machine translation models often produce overly formal, archaic, or bookish translations (e.g., using *कृपया*, *प्रसन्नता*, *दूरभाष यंत्र*). **IndicTrans2-SimpleHindi** bridges this gap by shifting the output distribution to natural modern Hindi with everyday idioms and natural Devanagari loanwords (e.g., *टेंशन*, *प्लान*, *ऑफिस*, *लाइटिंग*, *मीटिंग*).

---

## Key Features

- **Colloquial Everyday Hindi:** Outputs sound like natural conversational speech rather than textbook literature.
- **Natural Loanword Spelling:** Common modern terms are naturally transcribed in Devanagari (*ऑर्डर*, *टेंशन*, *कंट्रोल*).
- **Anti-Catastrophic Forgetting Mix:** Fine-tuned on an **80/20 balanced blend** (~4,500 everyday conversational lines + 1,000 standard benchmark pairs) so the model retains formal and long-sentence translation competence.
- **Lightweight LoRA Architecture:** Modifies attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) with only **3.5M trainable parameters (~1.64%)**, requiring only ~14 MB for the adapter weights.
- **Ready for CPU & GPU:** Auto-selects CUDA when available (with FP16 precision) and falls back cleanly to CPU.

---

## Benchmark Comparisons

| English Input | Default IndicTrans2 (Formal) | IndicTrans2-SimpleHindi (v0.1.1 beta) |
|---|---|---|
| *Don't worry about it, I will handle it.* | इसके बारे में चिंता मत करो, मैं इसे संभालूंगा। | चिंता मत करो, मैं संभाल लूंगा। |
| *Where are you going right now?* | आप अभी कहाँ जा रहे हैं? | तुम अभी कहाँ जा रहे हो? |
| *Don't take tension, everything is under control.* | तनाव मत लो, सब कुछ नियंत्रण में है। | टेंशन मत लो, सब कुछ कंट्रोल में है। |

---

## Quickstart

### 1. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/IndicTrans2-SimpleHindi.git
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
 IndicTrans2-SimpleHindi v0.1.1 (beta)
 Running on: CUDA
============================================================

Model ready! Enter English text to translate to everyday conversational Hindi.
Type 'q' or 'exit' to quit.

EN > How are you doing today?
HI > आज कैसा चल रहा है?

EN > Don't worry about it, I will handle it.
HI > चिंता मत करो, मैं संभाल लूंगा।
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

- **`daily_hindi_conversational.csv`**: Raw curated everyday conversational pairs.
- **`train_data.csv`**: 5,100 training pairs (4,180 conversational + 920 general benchmark pairs).
- **`val_data.csv`**: 400 held-out evaluation pairs (320 conversational + 80 general benchmark pairs).
- **`daily_hindi_mixed.csv`**: Unified 5,500-pair reference dataset.

To re-train or fine-tune with custom parameters:

```bash
python train.py
```

### Training Configuration
- **Base Model:** `ai4bharat/indictrans2-en-indic-dist-200M`
- **LoRA Parameters:** $r = 16, \alpha = 32, \text{dropout} = 0.05$
- **Learning Rate:** $1\times 10^{-4}$ with 25 warmup steps
- **Batch Size:** 16 (gradient accumulation 2, effective batch size 32)
- **Epochs:** 3
- **Validation Loss:** Decreased monotonically from `2.509` (Epoch 1) $\rightarrow$ `2.278` (Epoch 2) $\rightarrow$ **`2.207`** (Epoch 3).

---

## Project Structure

```
IndicTrans2-SimpleHindi/
├── model/                     # Pre-trained LoRA adapter weights (v0.1.1 beta)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── data/                      # Training and validation datasets
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

### v0.1.1 (beta)
- Initial public beta release.
- Pre-trained LoRA weights on 5,500 curated conversational and general mix pairs.
- Interactive `chat.py` CLI and `inference.py` helper.
- Full dataset and reproducibility scripts included.

---

## License & Attribution

- Built on top of [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) developed by AI4Bharat.
- Released under the MIT License.
