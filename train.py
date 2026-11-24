import inspect
import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from IndicTransToolkit.processor import IndicProcessor

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"

# 1. Load tokenizer, model, and IndicProcessor
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
ip = IndicProcessor(inference=False)

# 2. Configure / Load LoRA adapter (train on top of healthy v0.1.2 adapter)
adapter_dir = str(BASE_DIR / "model")
if (BASE_DIR / "model" / "adapter_model.safetensors").exists():
    print(f"Loading existing healthy adapter from {adapter_dir} for continuation fine-tuning (is_trainable=True)...")
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
else:
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# 3. Load Dataset
train_path = str(BASE_DIR / "data" / "train_data.csv")
val_path = str(BASE_DIR / "data" / "val_data.csv")
dataset = load_dataset("csv", data_files={"train": train_path, "test": val_path})

def preprocess_function(examples):
    inputs = ip.preprocess_batch(examples["en"], src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    targets = ip.preprocess_batch(examples["hi"], src_lang=TGT_LANG, is_target=True)

    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=False)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ds = dataset.map(preprocess_function, batched=True, remove_columns=["en", "hi"])

# 4. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=str(BASE_DIR / "model_checkpoints"),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    num_train_epochs=2,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    logging_steps=15,
    predict_with_generate=False,
)

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": tokenized_ds["train"],
    "eval_dataset": tokenized_ds["test"],
    "data_collator": DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
}
if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Seq2SeqTrainer(**trainer_kwargs)

if __name__ == "__main__":
    trainer.train()
    save_path = str(BASE_DIR / "model")
    model.save_pretrained(save_path)
    print(f"Fine-tuning completed. Adapter saved to {save_path}")
