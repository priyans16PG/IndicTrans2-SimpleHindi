---
base_model: ai4bharat/indictrans2-en-indic-dist-200M
library_name: peft
language:
- en
- hi
license: mit
tags:
- indictrans2
- lora
- peft
- machine-translation
- english-to-hindi
- simple-hindi
- conversational-hindi
- hinglish
- devanagari
- nlp
---

# IndicTrans2-SimpleHindi (`v1.0.1` LoRA Adapter)

A lightweight (~7 MB) LoRA adapter for [`ai4bharat/indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) fine-tuned to translate English into **natural, everyday spoken Hindi** instead of formal textbook translations.

## Model Details

- **Version:** `v1.0.1` (Stable)
- **Base Model:** `ai4bharat/indictrans2-en-indic-dist-200M`
- **Adapter Type:** PEFT LoRA (`r=8`, `lora_alpha=16`, `lora_dropout=0.05`)
- **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `out_proj`
- **Trainable Parameters:** `1,769,472` (`0.8286%` of `213,545,984` total parameters)
- **Languages:** English (`eng_Latn`) $\rightarrow$ Hindi (`hin_Deva`)
- **Repository:** [priyans16PG/IndicTrans2-SimpleHindi](https://github.com/priyans16PG/IndicTrans2-SimpleHindi)
- **License:** MIT

## Quickstart

> [!IMPORTANT]
> Pin `transformers==4.46.3` and `peft==0.13.2`. Newer `transformers 5.x` releases break IndicTrans2's custom `trust_remote_code` attention/cache logic.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
ADAPTER_PATH = "./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).merge_and_unload().to(DEVICE)
model.eval()

ip = IndicProcessor(inference=True)
batch = ip.preprocess_batch(
    ["Just take a screenshot of the error message and share it in the group."],
    src_lang="eng_Latn",
    tgt_lang="hin_Deva",
)
inputs = tokenizer(batch, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        use_cache=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
    )

decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
print(ip.postprocess_batch(decoded, lang="hin_Deva")[0])
# बस एरर मैसेज का स्क्रीन शॉट लें और ग्रुप में शेयर करें।
```

For automatic multi-sentence paragraph splitting and non-Devanagari token suppression, use `translate_batch()` from `inference.py` or the interactive CLI in `chat.py`.

## Training Summary (`v1.0.1`)

- **Foundation (`v1.0.0`):** `lr=5e-5`, cosine scheduler, `max_length=256`, `label_pad_token_id=-100` on conversational/tech pairs, narrative prose, and stitched 2-sentence pairs (`eval_loss: 1.1544`).
- **Patch (`v1.0.1`):** Gentle continuation (`lr=3e-5`, 2 epochs) on 200 modern tech/slang pairs + 250 `v1.0.0` replay pairs (`eval_loss: 1.1733`) to restore tech/workplace loanwords without losing narrative flow.

## Limitations

- Abstract self-help/finance terms may occasionally retain formal base-model vocabulary.
- A few figurative English idioms can still translate literally.
- For multi-sentence paragraphs, use sentence splitting (built into `inference.py` and `chat.py`).

## Attribution

Built on top of [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) by AI4Bharat.
