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
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(DEVICE)
model.eval()

# Precompute non-Devanagari / Arabic script token IDs to suppress
arabic_re = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
SUPPRESS_TOKENS = [tid for tid, token in tokenizer.tgt_decoder.items() if arabic_re.search(token)]

ip = IndicProcessor(inference=True)

def clean_output(text: str) -> str:
    text = re.sub(r'^(?:_[A-Za-z]+|[a-z]{3}_[A-Za-z]+)\s*', '', text)
    text = re.sub(r'^[\s"\'“”]+', '', text)
    text = re.sub(r'[\s"\'“”]+$', '', text)
    text = re.sub(r'["“”]{2,}', '', text)
    return text.strip()

def translate_batch(sentences: list[str]) -> list[str]:
    batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva")
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            use_cache=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            suppress_tokens=SUPPRESS_TOKENS,
        )

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    translations = ip.postprocess_batch(decoded, lang="hin_Deva")
    return [clean_output(t) for t in translations]

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
