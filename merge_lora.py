import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from safetensors.torch import save_file

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
ADAPTER_PATH = "./model"
MERGED_OUTPUT_DIR = "./merged_model"

def main():
    print("=" * 60)
    print("Step 1: Merging LoRA adapter with base IndicTrans2-200M model...")
    print("=" * 60)
    
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    
    print("Loading LoRA weights from", ADAPTER_PATH)
    lora_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print("Merging adapter weights...")
    merged_model = lora_model.merge_and_unload()
    merged_model.eval()
    
    print(f"Saving merged weights and config to {MERGED_OUTPUT_DIR}...")
    # Clone state_dict to break shared memory view between decoder.embed_tokens and lm_head
    state_dict = {k: v.clone() for k, v in merged_model.state_dict().items()}
    save_file(state_dict, os.path.join(MERGED_OUTPUT_DIR, "model.safetensors"))
    
    merged_model.config.save_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    print("Merged model saved successfully!")

if __name__ == "__main__":
    main()
