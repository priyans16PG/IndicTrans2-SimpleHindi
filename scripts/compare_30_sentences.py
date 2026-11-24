import os
import re
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
ADAPTER_PATH = str(REPO_ROOT / "model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE.upper()}")
print(f"Loading Base Model: {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(DEVICE)
base_model.eval()

# Precompute non-Devanagari / Arabic script token IDs to suppress
arabic_re = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
SUPPRESS_TOKENS = [tid for tid, token in tokenizer.tgt_decoder.items() if arabic_re.search(token)]

ip = IndicProcessor(inference=True)

test_set_file = SCRIPT_DIR / "simple_hindi_test_set.txt"
if not test_set_file.exists():
    test_set_file = REPO_ROOT / "simple_hindi_test_set.txt"
if not test_set_file.exists():
    test_set_file = REPO_ROOT.parent / "simple_hindi_test_set.txt"
with open(test_set_file, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(sentences)} test sentences.")

def clean_output(text: str) -> str:
    text = re.sub(r'^(?:[\d\-/.]*(?:_?[A-Za-z]+|\w+_Deva)[\d\-/.]*\s*|[\d\-/.]+\s*)+', '', text)
    text = re.sub(r'^[\d\-/.]+(?=[\u0900-\u097F])', '', text)
    text = re.sub(r'^[^\w\s\u0900-\u097F]+', '', text)
    text = re.sub(r'[\s"\'“”]+$', '', text)
    text = re.sub(r'["“”]{2,}', '', text)
    return text.strip()

def run_inference(model, sents, batch_size=6):
    outs = []
    for i in range(0, len(sents), batch_size):
        chunk = sents[i:i + batch_size]
        prep = ip.preprocess_batch(chunk, src_lang="eng_Latn", tgt_lang="hin_Deva")
        inputs = tokenizer(prep, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                use_cache=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                suppress_tokens=SUPPRESS_TOKENS,
            )
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        post = ip.postprocess_batch(dec, lang="hin_Deva")
        outs.extend([clean_output(t) for t in post])
    return outs

print("\nTranslating with BASE MODEL (Without LoRA)...")
base_results = run_inference(base_model, sentences)

print("Loading LoRA Adapter (v0.1.3)...")
lora_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(DEVICE)
lora_model.eval()

print("Translating with LoRA MODEL (With LoRA)...")
lora_results = run_inference(lora_model, sentences)

print("\n" + "=" * 90)
print(f"{'#':<3} | {'ENGLISH INPUT':<35} | {'BASE MODEL (WITHOUT LoRA)':<40} | {'LoRA (WITH LoRA - v0.1.3)':<40}")
print("=" * 90)

for idx, (en, base_hi, lora_hi) in enumerate(zip(sentences, base_results, lora_results), 1):
    print(f"\n[{idx:02d}] EN       : {en}")
    print(f"     BASE (No LoRA): {base_hi}")
    print(f"     LoRA (v0.1.3) : {lora_hi}")
