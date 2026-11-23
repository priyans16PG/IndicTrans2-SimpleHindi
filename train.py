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
from peft import LoraConfig, get_peft_model, TaskType
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"

# 1. Load tokenizer, model, and IndicProcessor
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.float16,
)
ip = IndicProcessor(inference=False)

# 2. Configure LoRA (targets attention projections)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Load Dataset
BASE_DIR = Path(__file__).resolve().parent
train_path = str(BASE_DIR / "data" / "train_data.csv")
val_path = str(BASE_DIR / "data" / "val_data.csv")
dataset = load_dataset("csv", data_files={"train": train_path, "test": val_path})

def preprocess_function(examples):
    inputs = ip.preprocess_batch(examples["en"], src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    targets = ip.preprocess_batch(examples["hi"], src_lang=TGT_LANG, tgt_lang=SRC_LANG)

    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=False)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ds = dataset.map(preprocess_function, batched=True, remove_columns=["en", "hi"])

# 4. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./model_checkpoints",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=25,
    weight_decay=0.01,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    logging_steps=25,
    predict_with_generate=False,
)

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": tokenized_ds["train"],
    "eval_dataset": tokenized_ds["test"],
    "data_collator": DataCollatorForSeq2Seq(tokenizer, model=model),
}
if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Seq2SeqTrainer(**trainer_kwargs)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("./model")
    print("Fine-tuning completed. Adapter saved to ./model")
