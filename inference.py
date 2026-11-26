import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
ADAPTER_PATH = os.path.join(BASE_DIR, "model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(DEVICE)
model.eval()

# Precompute non-Devanagari / Arabic script token IDs to suppress
arabic_re = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
SUPPRESS_TOKENS = [tid for tid, token in tokenizer.tgt_decoder.items() if arabic_re.search(token)]

ip = IndicProcessor(inference=True)

_DOT_PLACEHOLDER = "\x00DOT\x00"
_ABBREV_RE = re.compile(
    r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|e\.g|i\.e|etc|vs|Rs|No)\.',
    re.IGNORECASE,
)

def split_sentences(text: str) -> list[list[str]]:
    """Split text into paragraphs (newline-separated) and individual sentences,
    preserving common abbreviations and ellipses."""
    paragraphs = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            paragraphs.append([])
            continue
        protected = line.replace("...", _DOT_PLACEHOLDER * 3)
        protected = _ABBREV_RE.sub(lambda m: m.group(0)[:-1] + _DOT_PLACEHOLDER, protected)
        parts = re.split(r'(?<=[.!?]["”’\'\)])\s+|(?<=[.!?])\s+', protected)
        sents = [p.replace(_DOT_PLACEHOLDER, ".").strip() for p in parts if p.strip()]
        paragraphs.append(sents)
    return paragraphs

def clean_output(text: str) -> str:
    text = re.sub(r'^(?:[\d\-/.]*(?:_?[A-Za-z]+|\w+_Deva)[\d\-/.]*\s*|[\d\-/.]+\s*)+', '', text)
    text = re.sub(r'^[\d\-/.]+(?=[\u0900-\u097F])', '', text)
    text = re.sub(r'^[^\w\s\u0900-\u097F]+', '', text)
    text = re.sub(r'[\s"\'“”]+$', '', text)
    text = re.sub(r'["“”]{2,}', '', text)
    return text.strip()

def _translate_flat(sentences: list[str], batch_size: int = 8) -> list[str]:
    if not sentences:
        return []
    results = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        batch = ip.preprocess_batch(chunk, src_lang="eng_Latn", tgt_lang="hin_Deva")
        inputs = tokenizer(batch, truncation=True, max_length=256, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                use_cache=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                suppress_tokens=SUPPRESS_TOKENS,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations = ip.postprocess_batch(decoded, lang="hin_Deva")
        results.extend([clean_output(t) for t in translations])
    return results

def translate_batch(sentences: list[str]) -> list[str]:
    """Translate a list of English strings (sentences or multi-sentence paragraphs),
    automatically splitting into sentences for GPU batching and reassembling paragraphs."""
    structured = [split_sentences(text) for text in sentences]
    flat_sents = [
        sent
        for para_list in structured
        for line_sents in para_list
        for sent in line_sents
    ]
    translated_flat = _translate_flat(flat_sents)

    outputs = []
    idx = 0
    for para_list in structured:
        line_outputs = []
        for line_sents in para_list:
            count = len(line_sents)
            if count == 0:
                line_outputs.append("")
            else:
                line_outputs.append(" ".join(translated_flat[idx:idx + count]))
                idx += count
        outputs.append("\n".join(line_outputs))
    return outputs

if __name__ == "__main__":
    sample_queries = [
        "Don't worry about it, I will handle it.",
        "Where are you going right now?",
        "Can you send me the location on WhatsApp?",
    ]
    results = translate_batch(sample_queries)
    for en, hi in zip(sample_queries, results):
        print(f"EN: {en}")
        print(f"HI: {hi}\n")
