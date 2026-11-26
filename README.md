# IndicTrans2-SimpleHindi

> **Version:** `v1.0.0 (Stable)`
>
> **Base Model:** [`ai4bharat/indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
>
> **Status:** Stable — Verified on 30-sentence standard suite, 27-item stress test, and 30-item v3 benchmark (including multi-sentence paragraphs)

**IndicTrans2-SimpleHindi** is a lightweight LoRA adapter (\~7 MB) for IndicTrans2 200M fine-tuned to produce **natural, everyday spoken Hindi** instead of stiff, textbook translations.

Standard English-to-Hindi models often output archaic or overly formal vocabulary (e.g., *यातायात*, *बैठक*, *समय सीमा*, *साक्षात्कार*, *मात्रात्मक*). **IndicTrans2-SimpleHindi** shifts the output toward modern conversational Hindi with clean Devanagari loanwords (e.g., *ट्रैफिक*, *मीटिंग*, *डेडलाइन*, *इंटरव्यू*, *क्वांटाइज्ड*, *वॉट्सऐप*) while preserving multi-clause logic, storytelling flow, and paragraph completeness.

## Key Features

* **Natural Spoken Hindi:** Replaces formal textbook phrasing with everyday conversational Hindi and smooth narrative prose.

* **Paragraph-Safe Sentence Splitting:** Automatically splits multi-sentence paragraphs in `inference.py` and `chat.py` (preserving abbreviations like `Mr.`, `Mrs.`, `Dr.`, `e.g.`, `...`, and newline breaks) so long paragraphs never drop middle sentences.

* **Modern Tech & Indian Context:** Handles loanwords naturally in Devanagari (*अपडेट इंस्टॉल*, *एरर मैसेज*, *डिलीवरी बॉय*, *मेन गेट*, *फ्लाइट मिस*, *क्वांटाइज्ड मॉडल*).

* **Fixes Base Literalisms:** Corrects common base-model slips (e.g., *call me back* → *वापस कॉल कर सकते हो* instead of *वापस बुला सकते हैं*; *lag issue* → *लैग की समस्या* instead of *अंतराल की समस्या*).

* **Strong Multi-Clause Logic:** 0% clause-swapping and 0% repetition loops across both the 30-sentence standard suite (**29/30 accuracy**) and the 27-item stress test (including 30–40 word sentences and double negatives).

* **Non-Devanagari Token Suppression:** Dynamically suppresses 19k+ non-Devanagari tokens from IndicTrans2's shared 22-language vocabulary during generation to prevent cross-script leakage.

* **Lightweight LoRA Adapter:** Only **1.77M trainable parameters (\~0.83%)** on attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`), weighing \~7 MB on disk.

* **CPU & GPU Ready:** Automatically uses CUDA (FP16) when available and falls back cleanly to CPU.

## Side-by-Side Comparison (Base vs. v1.0.0)

| Category | English Input | Base IndicTrans2 (Formal) | IndicTrans2-SimpleHindi (`v1.0.0`) |
| ----- | ----- | ----- | ----- |
| **Daily Chat** | *No worries at all, we can easily reschedule for tomorrow evening.* | चिंता की कोई बात नहीं, हम आसानी से कल शाम के लिए पुनर्निर्धारित कर सकते हैं। | कोई चिंता नहीं, हम आसानी से कल शाम के लिए रीशेड्यूल कर सकते हैं। |
| **Tech / PC** | *Always keep a backup of your project folder on Google Drive before formatting your PC.* | अपने पीसी को प्रारूपित करने से पहले हमेशा गूगल ड्राइव पर अपने परियोजना फ़ोल्डर का एक बैकअप रखें। | अपने पीसी को फॉर्मेट करने से पहले हमेशा गूगल ड्राइव पर अपने प्रोजेक्ट फ़ोल्डर का बैकअप रखें। |
| **Mindset** | *If you wait until you feel completely ready, you will probably keep postponing your dream forever.* | यदि आप तब तक प्रतीक्षा करते हैं जब तक कि आप पूरी तरह से तैयार महसूस नहीं करते हैं, तो आप शायद अपने सपने को हमेशा के लिए स्थगित करते रहेंगे। | अगर आप पूरी तरह से तैयार महसूस करने तक इंतजार करते हैं, तो आप शायद अपना सपना हमेशा के लिए टालते रहेंगे। |
| **Indian Context** | *We ordered biryani from Zomato, and the delivery guy got lost near the main gate.* | हमने जोमैटो से बिरयानी मंगाई, और डिलीवरी करने वाला मुख्य द्वार के पास खो गया। | हमने जोमैटो से बिरयानी मंगाई, और डिलीवरी बॉय मेन गेट के पास खो गया। |
| **Tech / AI** | *Now, with quantized weights and ONNX runtimes, you can translate an entire textbook chapter on a train with zero internet connection.* | अब, मात्रात्मक भार और ओएनएनएक्स रनटाइम के साथ, आप शून्य इंटरनेट कनेक्शन के साथ एक ट्रेन में एक पूरे पाठ्यपुस्तक अध्याय का अनुवाद कर सकते हैं। | अब, क्वांटाइज्ड वेट और ओ. एन. एक्स. रनटाइम के साथ, आप शून्य इंटरनेट कनेक्शन वाली ट्रेन में एक पूरे पाठ्यपुस्तक अध्याय का अनुवाद कर सकते हैं। |

## Quickstart

### 1. Installation

```
git clone https://github.com/priyans16PG/IndicTrans2-SimpleHindi.git
cd IndicTrans2-SimpleHindi
pip install -r requirements.txt

```

> \[!IMPORTANT\]
> **Pin `transformers==4.46.3`:** You **must** keep `transformers==4.46.3`, `peft==0.13.2`, `datasets==3.6.0`, and `huggingface-hub==0.36.2` as specified in `requirements.txt`. Do **not** upgrade to `transformers 5.x` — breaking changes to attention masks and KV-cache handling in 5.x corrupt IndicTrans2's `trust_remote_code` model, resulting in scrambled "bag-of-words" translations even on the untouched base model.

*(Make sure you have accepted the model terms for `ai4bharat/indictrans2-en-indic-dist-200M` on Hugging Face and authenticated via `huggingface-cli login`.)*

### 2. Interactive CLI Translator

```
python chat.py

```

```
============================================================
 IndicTrans2-SimpleHindi v1.0.0
 Running on: CUDA
============================================================

EN > If you don't start working on the project today, you won't finish before the deadline.
HI > अगर तुम आज प्रोजेक्ट पर काम शुरू नहीं करोगे, तो तुम डेडलाइन से पहले खत्म नहीं करोगे।

EN > Just take a screenshot of the error message and share it in the group.
HI > बस एरर मैसेज का स्क्रीन शॉट लें और ग्रुप में शेयर करें।

```

### 3. Python Batch Inference

```
from inference import translate_batch

sentences = [
    "Don't worry about it, I will handle everything.",
    "The new app update fixed the lag issue, and now the interface feels much smoother.",
    "He closed his laptop, picked up his keys, and walked out of the office without saying a word.",
]

translations = translate_batch(sentences)
for src, tgt in zip(sentences, translations):
    print(f"EN: {src}\nHI: {tgt}\n")

```

> \[!TIP\]
> **Translating Long Paragraphs or Book Chapters:** Like the base IndicTrans2 model, this adapter performs best on single sentences. Feeding raw multi-sentence paragraphs (3+ long sentences at once) can occasionally cause the model to skip a middle sentence. Always split paragraphs into individual sentences first (e.g., with `nltk.sent_tokenize` or regex), pass the list to `translate_batch()`, and join the results back together.

### 4. Running the Benchmarks

```bash
# Run the 30-sentence standard evaluation suite
python scripts/run_30_benchmark.py

# Compare Base vs. LoRA side-by-side
python scripts/debug_compare.py

# Run the 27-item stress-test benchmark
python scripts/run_benchmark_v2.py
```

## Training & Evaluation Summary

| Metric / Setting | Value | 
| ----- | ----- | 
| **Base Model** | `ai4bharat/indictrans2-en-indic-dist-200M` | 
| **LoRA Config** | `r=8`, `lora_alpha=16`, `lora_dropout=0.05` | 
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `out_proj` (1.77M params / \~0.83%) | 
| **Training Strategy** | Continuation fine-tune from `v0.1.2` (`is_trainable=True`, cosine schedule, 2 epochs) | 
| **Continuation Dataset** | 1,940 train / 200 val curated pairs (book dialogues, chat, modern loanwords) | 
| **Validation Loss** | **`1.312`** (down from `1.927` in `v0.1.2` and `2.207` in `v0.1.1`) | 
| **Decoding Safeguards** | `no_repeat_ngram_size=3`, `repetition_penalty=1.2`, non-Devanagari `suppress_tokens` | 
| **30-Sentence Benchmark** | **29/30 accurate**, 30/30 coherent (0% clause-swapping, 0% script leaks) | 
| **27-Item Stress Test** | Outperforms Base on **23/27 items** across chat, tech, idioms, and long sentences | 

To re-train or continue fine-tuning:

```
python train.py

```

## Project Structure

```
IndicTrans2-SimpleHindi/
├── model/                          # Fine-tuned LoRA adapter weights (v1.0.0 Stable)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── chat.py                         # Interactive terminal translator (with sentence splitting)
├── inference.py                    # Batch & paragraph inference helper
├── train.py                        # LoRA fine-tuning script
├── requirements.txt                # Pinned dependencies (transformers==4.46.3)
├── .gitignore
└── README.md
```

## Known Limitations & Roadmap

### Current Limitations (`v1.0.0`)

* **Non-Fiction / Mindset Vocabulary:** Abstract self-help and finance terms still occasionally retain formal base-model vocabulary (e.g., *प्रतिभाशाली*, *व्यवस्थाएँ*, *तत्काल परिणाम*, *चक्रवृद्धि*).

* **English Idioms:** A few figurative idioms (*"too much on his plate"*, *"acting up"*) can still translate literally.


## Release Notes

### `v1.0.0` (Stable)

* **First Stable Release:** Graduated from beta after passing 30-sentence standard, 27-item stress-test, and 30-item v3 benchmarks.

* **Enhanced Storytelling & Non-Fiction Fluency:** Fine-tuned on modern conversational prose and narrative pairs while retaining everyday tech and workplace loanwords.

* **Built-In Paragraph Sentence Splitting:** Automatic `split_sentences()` pipeline in `inference.py` and `chat.py` (preserving abbreviations and `\n` breaks) for reliable multi-sentence paragraph translation.

* **Lean Repository Structure:** Clean, self-contained repository tracking only core inference, adapter weights, and training scripts.

### `v0.1.4` (beta)

* **Paragraph Sentence-Splitting Pipeline:** Added `split_sentences()` to `inference.py` and `chat.py` (preserving abbreviations like `Mr.`, `Mrs.`, `Dr.`, `St.`, `e.g.`, `i.e.`, `...`, and `\n` breaks), eliminating dropped middle sentences in multi-sentence paragraphs.

* **Narrative & Storytelling Continuation:** Fine-tuned gently on top of `v0.1.3` (`lr=5e-5`, 2 epochs, `max_length=256`) using cleaned *Wise and Otherwise* parallel pairs, 10% stitched 2-sentence pairs, and a 450-row conversational/tech replay buffer — reducing validation loss from `1.312` to **`1.154`**.

* **Clean Core Repository:** Untracked internal datasets, dev scripts, and benchmark files via `.gitignore` so the repository tracks only core user-facing files.

### `v0.1.3` (beta)

* **Fixed `transformers 5.x` Attention Bug:** Pinned `transformers==4.46.3` and `peft==0.13.2` after isolating upstream attention/cache breakage that scrambled translations.

* **Targeted Continuation Fine-Tuning:** Trained on 2,140 curated pairs, dropping validation loss from `1.927` to **`1.312`**.

* **Vocabulary Modernization:** Shifted formal terms to everyday Hindi (*यदि* → *अगर*, *समय सीमा* → *डेडलाइन*, *यातायात* → *ट्रैफिक*, *बैठक* → *मीटिंग*, *स्थान* → *लोकेशन*).

* **Cross-Script Protection:** Added automatic suppression of 19k+ non-Devanagari tokens during decoding.

### `v0.1.2` (beta)

* Merged Tatoeba English–Hindi pairs (6,072 lines), expanding the dataset to 12,913 pairs.

* Reduced validation loss from `2.207` to `1.927`.

### `v0.1.1` (beta)

* Initial public beta LoRA adapter trained on 5,500 conversational pairs, with `chat.py` and `inference.py`.

## License & Attribution

* Built on top of [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) by AI4Bharat.

* Released under the **MIT License**.