import os
import sys
import re
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
ADAPTER_PATH = os.path.join(BASE_DIR, "model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print(" IndicTrans2-SimpleHindi v1.0.0")
print(f" Running on: {DEVICE.upper()}")
print("=" * 60)

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

def translate(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    structured = split_sentences(text)
    flat_sents = [sent for line_sents in structured for sent in line_sents]
    if not flat_sents:
        return ""

    batch = ip.preprocess_batch(flat_sents, src_lang="eng_Latn", tgt_lang="hin_Deva")
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
    translations = [clean_output(t) for t in ip.postprocess_batch(decoded, lang="hin_Deva")]

    line_outputs = []
    idx = 0
    for line_sents in structured:
        count = len(line_sents)
        if count == 0:
            line_outputs.append("")
        else:
            line_outputs.append(" ".join(translations[idx:idx + count]))
            idx += count
    return "\n".join(line_outputs)

def main():
    print("\nModel ready! Enter English text to translate to everyday conversational Hindi.")
    print("Type 'q' or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("EN > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "quit", "exit"]:
                print("Exiting.")
                break

            output_hi = translate(user_input)
            print(f"HI > {output_hi}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
