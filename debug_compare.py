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

print(f"Device: {DEVICE.upper()}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(DEVICE)
base_model.eval()

ip = IndicProcessor(inference=True)

test_sentences = [
    "I am stuck in traffic right now, so I will be ten minutes late for the meeting.",
    "Don't waste your time worrying about things you cannot control.",
    "My phone battery died, so I couldn't check your message on WhatsApp.",
    "Small habits may not seem important today, but over a year they make a huge difference.",
    "He closed his laptop, picked up his keys, and walked out of the office without saying a word.",
]

def clean_output(text: str) -> str:
    text = re.sub(r'^(?:_[A-Za-z]+|[a-z]{3}_[A-Za-z]+)\s*', '', text)
    text = re.sub(r'^[\s"\'“”]+', '', text)
    text = re.sub(r'[\s"\'“”]+$', '', text)
    text = re.sub(r'["“”]{2,}', '', text)
    return text.strip()

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
