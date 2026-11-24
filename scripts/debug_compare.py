import os
import sys
import re
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
ADAPTER_PATH = str(REPO_ROOT / "model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE.upper()}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(DEVICE)
base_model.eval()

ip = IndicProcessor(inference=True)

test_sentences = [
    "Small habits may not seem important today, but over a year they make a huge difference.",
    "If you don't start working on the project today, you won't finish before the deadline.",
    "When you remove distractions from your desk, it becomes much easier to focus on deep work.",
    "After searching the entire house for an hour, he realized his glasses were on his head the whole time.",
]

def clean_output(text: str) -> str:
    text = re.sub(r'^(?:[\d\-/.]*(?:_?[A-Za-z]+|\w+_Deva)[\d\-/.]*\s*|[\d\-/.]+\s*)+', '', text)
    text = re.sub(r'^[\d\-/.]+(?=[\u0900-\u097F])', '', text)
    text = re.sub(r'^[^\w\s\u0900-\u097F]+', '', text)
    text = re.sub(r'[\s"\'“”]+$', '', text)
    text = re.sub(r'["“”]{2,}', '', text)
    return text.strip()

arabic_re = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
SUPPRESS_TOKENS = [tid for tid, token in tokenizer.tgt_decoder.items() if arabic_re.search(token)]

def run_inference(model, sentences):
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
    raw = ip.postprocess_batch(decoded, lang="hin_Deva")
    return [clean_output(t) for t in raw]

print("\n--- Running Base Model ---")
base_outputs = run_inference(base_model, test_sentences)

print("\n--- Loading LoRA Adapter ---")
lora_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(DEVICE)
lora_model.eval()

print("--- Running LoRA Model ---")
lora_outputs = run_inference(lora_model, test_sentences)

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPARISON: BASE MODEL vs. CURRENT LoRA ADAPTER")
print("=" * 80)
for idx, (sent, b_out, l_out) in enumerate(zip(test_sentences, base_outputs, lora_outputs), 1):
    print(f"\n[Sentence {idx}]: {sent}")
    print(f"  BASE : {b_out}")
    print(f"  LoRA : {l_out}")
print("=" * 80 + "\n")
