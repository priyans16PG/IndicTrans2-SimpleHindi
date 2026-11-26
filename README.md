<div align="center">

# IndicTrans2-SimpleHindi

**Lightweight LoRA Adapter for Natural, Everyday English-to-Simple-Hindi Translation**

[![Release](https://img.shields.io/github/v/release/priyans16PG/IndicTrans2-SimpleHindi?color=blue&label=Release)](https://github.com/priyans16PG/IndicTrans2-SimpleHindi/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Base Model](https://img.shields.io/badge/Base%20Model-IndicTrans2%20200M-orange)](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
[![LoRA Size](https://img.shields.io/badge/Adapter%20Size-~7%20MB-purple)](./model)
[![Transformers](https://img.shields.io/badge/transformers-4.46.3-yellow)](./requirements.txt)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

*Turn stiff, textbook machine translations into natural, everyday spoken Hindi — offline, fast, and in clean Devanagari script.*

[Quickstart](#-quickstart) • [Side-by-Side Comparison](#-side-by-side-comparison-base-vs-v100-vs-v101) • [Use Cases](#-use-cases) • [Training & Benchmarks](#-training--evaluation-summary) • [Release Notes](#-release-notes)

</div>

---

## Why IndicTrans2-SimpleHindi?

Standard English-to-Hindi neural machine translation (NMT) models—including the base [`ai4bharat/indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)—often generate overly formal, archaic, or Sanskritized textbook Hindi (e.g., *यातायात*, *बैठक*, *समय सीमा*, *त्रुटि संदेश*, *मात्रात्मक*). While grammatically correct, that style feels unnatural in **spoken Hindi**, **books**, **YouTube/movie subtitles**, **chat apps**, and **everyday Indian tech contexts**.

**IndicTrans2-SimpleHindi (`v1.0.1 Stable`)** is a compact **~7 MB PEFT LoRA adapter** (`1.77M` trainable parameters, `0.83%` of base) that steers IndicTrans2 200M toward **natural conversational Hindi** and **modern Devanagari loanwords** (e.g., *ट्रैफिक*, *मीटिंग*, *डेडलाइन*, *एरर मैसेज*, *ग्रुप में शेयर*, *क्वांटाइज्ड मॉडल*, *वॉट्सऐप*) while preserving multi-clause grammar, negation, and storytelling nuance.

---

## ✨ Key Features

- **Natural Conversational Hindi (Not Textbook Hindi):** Replaces stiff phrasing (*दोपहर का भोजन*, *अद्यतन स्थापित*) with everyday speech (*लंच*, *अपडेट इंस्टॉल*).
- **Zero-Overhead Merged Inference:** `chat.py` uses `merge_and_unload()` + `use_cache=True` for fast CUDA (FP16) and CPU inference without LoRA runtime overhead.
- **Paragraph-Safe Sentence Splitting (`split_sentences`):** Automatically splits multi-sentence inputs and book paragraphs across `.`, `?`, `!`, and `\n` breaks while protecting abbreviations (`Mr.`, `Mrs.`, `Dr.`, `St.`, `e.g.`, `i.e.`, `...`), preventing skipped middle sentences.
- **Modern Tech & Indian Workplace Vocabulary:** Naturally renders everyday English terms in Devanagari (*एरर मैसेज*, *वर्क कल्चर*, *क्वांटाइज्ड मॉडल*, *डिलीवरी बॉय*, *मेन गेट*, *फ्लाइट मिस*, *रीशेड्यूल*).
- **Fixes Base-Model Literalisms:** Resolves awkward literal translations (e.g., *call me back* → *वापस कॉल कर सकते हो* instead of *वापस बुला सकते हैं*; *lag issue* → *लैग की समस्या* instead of *अंतराल की समस्या*).
- **Non-Devanagari Token Suppression:** Dynamically masks 19,000+ non-Devanagari / Arabic-script tokens from IndicTrans2's shared 22-language vocabulary during decoding to guarantee 0% cross-script leakage.
- **Tiny Footprint, Local & Private:** Runs completely offline on consumer GPUs (6 GB VRAM+) or CPU with a ~7 MB adapter.

---

## 🔍 Side-by-Side Comparison (Base vs. `v1.0.0` vs. `v1.0.1`)

| Category | English Input | Base IndicTrans2 (Formal) | Previous (`v1.0.0`) | Current (`v1.0.1`) |
| --- | --- | --- | --- | --- |
| **Daily Chat** | *No worries at all, we can easily reschedule for tomorrow evening.* | चिंता की कोई बात नहीं, हम आसानी से कल शाम के लिए पुनर्निर्धारित कर सकते हैं। | कोई चिंता नहीं, हम आसानी से कल शाम के लिए रीशेड्यूल कर सकते हैं। | बिल्कुल भी चिंता मत करो, हम आसानी से कल शाम के लिए रीशेड्यूल कर सकते हैं। |
| **Tech / PC** | *Always keep a backup of your project folder on Google Drive before formatting your PC.* | अपने पीसी को प्रारूपित करने से पहले हमेशा गूगल ड्राइव पर अपने परियोजना फ़ोल्डर का एक बैकअप रखें। | अपने पीसी को फॉर्मेट करने से पहले हमेशा गूगल ड्राइव पर अपने प्रोजेक्ट फ़ोल्डर का बैकअप रखें। | अपने पीसी को फॉर्मेट करने से पहले हमेशा गूगल ड्राइव पर अपने प्रोजेक्ट फ़ोल्डर का बैकअप रखें। |
| **Tech / Chat** | *Just take a screenshot of the error message and share it in the group.* | बस त्रुटि संदेश का एक स्क्रीनशॉट लें और इसे समूह में साझा करें। | बस त्रुटि संदेश का एक स्क्रीन शॉट लें और उसे समूह में साझा करें। | बस एरर मैसेज का स्क्रीन शॉट लें और ग्रुप में शेयर करें। |
| **Mindset** | *If you wait until you feel completely ready, you will probably keep postponing your dream forever.* | यदि आप तब तक प्रतीक्षा करते हैं जब तक कि आप पूरी तरह से तैयार महसूस नहीं करते हैं, तो आप शायद अपने सपने को हमेशा के लिए स्थगित करते रहेंगे। | अगर आप पूरी तरह से तैयार महसूस करने तक इंतजार करते हैं, तो आप शायद अपना सपना हमेशा के लिए टालते रहेंगे। | अगर आप पूरी तरह से तैयार महसूस करने तक इंतजार करते हैं, तो आप शायद अपने सपने को हमेशा के लिए टालते रहेंगे। |
| **Indian Context** | *We ordered biryani from Zomato, and the delivery guy got lost near the main gate.* | हमने जोमैटो से बिरयानी मंगाई, और डिलीवरी करने वाला मुख्य द्वार के पास खो गया। | हमने जोमैटो से बिरयानी मंगाई, और डिलीवरी बॉय मेन गेट के पास खो गया। | हमने जोमैटो से बिरयानी मंगवाई, और डिलीवरी बॉय मेन गेट के पास खो गया। |
| **Tech / AI** | *Now, with quantized weights and ONNX runtimes, you can translate an entire textbook chapter on a train with zero internet connection.* | अब, मात्रात्मक भार और ओएनएनएक्स रनटाइम के साथ, आप शून्य इंटरनेट कनेक्शन के साथ एक ट्रेन में एक पूरे पाठ्यपुस्तक अध्याय का अनुवाद कर सकते हैं। | अब, क्वांटाइज्ड वेट और ओ. एन. एक्स. रनटाइम के साथ, आप शून्य इंटरनेट कनेक्शन वाली ट्रेन में एक पूरे पाठ्यपुस्तक अध्याय का अनुवाद कर सकते हैं। | अब, क्वांटाइज्ड वेट और ओएनएनएक्स रनटाइम के साथ, आप जीरो इंटरनेट कनेक्शन वाली ट्रेन में एक पूरे पाठ्यपुस्तक चैप्टर का अनुवाद कर सकते हैं। |

---

## 🎯 Use Cases

- **English Book & PDF Translation:** Translate non-fiction, self-help, and fiction paragraphs into fluent Hindi that reads naturally instead of sounding machine-translated.
- **Offline Voice Assistants & STT/TTS Pipelines:** Convert English transcripts into spoken-style Hindi ready for Indian TTS voices.
- **Subtitles, EdTech & Content Localization:** Localize tutorials, course scripts, and tech UIs using familiar Hinglish vocabulary in Devanagari.
- **Private Local AI Translation:** Zero cloud API costs or data leaks—runs entirely offline on your own hardware.

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/priyans16PG/IndicTrans2-SimpleHindi.git
cd IndicTrans2-SimpleHindi
pip install -r requirements.txt
```

> [!IMPORTANT]
> **Pin `transformers==4.46.3`:** Keep `transformers==4.46.3`, `peft==0.13.2`, `datasets==3.6.0`, and `huggingface-hub==0.36.2` as specified in `requirements.txt`. Do **not** upgrade to `transformers 5.x` — upstream changes to attention masks and KV-cache handling in 5.x break IndicTrans2's custom `trust_remote_code` model, producing scrambled translations even on the raw base model.

*(Ensure you have accepted the model license for [`ai4bharat/indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) on Hugging Face and logged in via `huggingface-cli login`.)*

### 2. Interactive CLI Translator

```bash
python chat.py
```

```text
============================================================
 IndicTrans2-SimpleHindi v1.0.1
 Running on: CUDA
============================================================

EN > If you don't start working on the project today, you won't finish before the deadline.
HI > अगर तुम आज प्रोजेक्ट पर काम शुरू नहीं करोगे, तो तुम डेडलाइन से पहले खत्म नहीं करोगे।

EN > Just take a screenshot of the error message and share it in the group.
HI > बस एरर मैसेज का स्क्रीन शॉट लें और ग्रुप में शेयर करें।
```

### 3. Python Batch & Paragraph Inference

`translate_batch()` handles both single sentences and multi-sentence paragraphs automatically:

```python
from inference import translate_batch

texts = [
    "Don't worry about it, I will handle everything.",
    "Dr. Sharma told Mr. Verma to take the tablet twice a day. However, Mr. Verma forgot the instructions at the pharmacy.",
]

translations = translate_batch(texts)
for src, tgt in zip(texts, translations):
    print(f"EN: {src}\nHI: {tgt}\n")
```

### 4. Fine-Tuning Your Own Adapter

Provide `data/train_data.csv` and `data/val_data.csv` with `en,hi` columns and run:

```bash
python train.py
```

---

## 📊 Training & Evaluation Summary

| Metric / Setting | Value |
| --- | --- |
| **Base Model** | `ai4bharat/indictrans2-en-indic-dist-200M` |
| **LoRA Architecture** | `r=8`, `lora_alpha=16`, `lora_dropout=0.05` (`q_proj`, `k_proj`, `v_proj`, `out_proj`) |
| **Trainable Parameters** | `1,769,472` (`0.8286%` of `213.5M` total parameters) |
| **Max Sequence Length** | `256` tokens (with `-100` label pad masking & `is_target=True`) |
| **Validation Loss Progression** | `2.207` (`v0.1.1`) → `1.927` (`v0.1.2`) → `1.312` (`v0.1.3`) → `1.154` (`v1.0.0`) → **`1.173`** (`v1.0.1` tech patch) |
| **Decoding Safeguards** | `num_beams=4`, `no_repeat_ngram_size=3`, `repetition_penalty=1.2`, non-Devanagari `suppress_tokens` |
| **Multi-Suite Verification** | Tested across 30-sentence suite (**29/30**), 27-item stress test, and 30-item v3 paragraph benchmark |

---

## 📁 Project Structure

```text
IndicTrans2-SimpleHindi/
├── model/                          # Fine-tuned LoRA adapter weights (v1.0.1 Stable)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md                   # PEFT Model Card
├── chat.py                         # Interactive terminal translator (merged weights + sentence splitting)
├── inference.py                    # Batch & paragraph Python translation API
├── train.py                        # LoRA fine-tuning script
├── requirements.txt                # Pinned compatible dependencies (transformers==4.46.3)
├── .gitignore
└── README.md
```

---

## ⚠️ Known Limitations

- **Abstract Non-Fiction / Finance Terms:** Domain-heavy terms occasionally retain formal base vocabulary (e.g., *प्रतिभाशाली*, *व्यवस्थाएँ*, *चक्रवृद्धि*).
- **Figurative English Idioms:** A small number of idioms (e.g., *"too much on his plate"*, *"acting up"*) may still translate literally.

---

## 📝 Release Notes

### `v1.0.1` (Stable)
- **Modern Tech & Workplace Loanword Refresh:** Gentle continuation patch (`lr=3e-5`, 2 epochs) on 200 modern tech/slang pairs + 250 `v1.0.0` replay pairs (`eval_loss=1.1733`). Restored everyday tech terms (*एरर मैसेज*, *ग्रुप में शेयर*, *वर्क कल्चर*, *क्वांटाइज्ड मॉडल*, *प्राइवेट डेटा*, *कंज्यूमर जीपीयू*, *वर्क सेव*) while preserving `v1.0.0`'s paragraph flow and pronoun consistency.

### `v1.0.0` (Stable)
- **First Stable Release:** Graduated from beta after passing the 30-sentence standard suite, 27-item stress test, and 30-item v3 benchmark.
- **Enhanced Storytelling & Non-Fiction Fluency:** Fine-tuned on modern conversational prose and narrative pairs while retaining everyday tech and workplace loanwords.
- **Built-In Paragraph Sentence Splitting:** Automatic `split_sentences()` pipeline in `inference.py` and `chat.py` (preserving abbreviations and `\n` breaks) for reliable multi-sentence paragraph translation.
- **Lean Repository Structure:** Clean, self-contained repository tracking only core inference, adapter weights, and training scripts.

<details>
<summary>Previous Beta Releases (v0.1.1 – v0.1.4)</summary>

- **`v0.1.4` (beta):** Added `split_sentences()` paragraph pipeline and trained narrative continuation with cleaned prose + stitched 2-sentence pairs (`eval_loss: 1.154`).
- **`v0.1.3` (beta):** Diagnosed and pinned `transformers==4.46.3` (fixing Transformers 5.x attention breakage), suppressed 19k+ non-Devanagari tokens, and trained on 2,140 curated pairs (`eval_loss: 1.312`).
- **`v0.1.2` (beta):** Merged Tatoeba English–Hindi pairs (6,072 lines) expanding dataset to 12,913 pairs (`eval_loss: 1.927`).
- **`v0.1.1` (beta):** Initial public beta LoRA adapter trained on 5,500 conversational pairs with `chat.py` and `inference.py`.

</details>

---

## 📄 License & Attribution

- Built on top of [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) developed by **AI4Bharat**.
- Released under the **MIT License**.
