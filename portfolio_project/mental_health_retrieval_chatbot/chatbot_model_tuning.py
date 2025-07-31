import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch

from constants import (
    DEFAULT_BASE_CHAT_MODEL,
    DEFAULT_CHAT_MODEL,
    EMPATHETIC_DIALOGUES_PATH,
    DIALOGPT_JSON_PATH,
    Names,
)
import pandas as pd
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
# -------------------
# Tokenizer + Model
# -------------------
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_CHAT_MODEL)

# add __eou__ token if not present
special_tokens = {"additional_special_tokens": [Names.EOU]}
tokenizer.add_special_tokens(special_tokens)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(DEFAULT_BASE_CHAT_MODEL)
model.resize_token_embeddings(len(tokenizer))  # Important if you add special tokens


# 1. Load EmpatheticDialogues
def load_empathetic(path: str = EMPATHETIC_DIALOGUES_PATH):
    df = pd.read_csv(path, quoting=1, on_bad_lines="skip")
    df[Names.TEXT] = df["prompt"] + " " + df["utterance"]
    return Dataset.from_pandas(df[[Names.TEXT]])


# -------------------
# Preprocessing
# -------------------
def filter_too_long(example):
    tokens = tokenizer(example[Names.TEXT], add_special_tokens=False)["input_ids"]
    return len(tokens) <= tokenizer.model_max_length

def tokenize_function(example):
    return tokenizer(
        example[Names.TEXT],
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

# -------------------
# Main Training
# -------------------
def main():
    # Load dataset
    ds = load_empathetic()#load_jsonl_dialogs(DIALOGPT_JSON_PATH)
    print("👉 Loaded dataset size:", len(ds))
    print("👉 Example:", ds["text"][0][:200], "...")

    # Shuffle + filter long dialogs
    ds = ds.shuffle(seed=42)
    ds = ds.filter(filter_too_long)
    print("👉 Filtered dataset size:", len(ds))

    # Tokenize
    tokenized = ds.map(
        tokenize_function, batched=True, remove_columns=[Names.TEXT]
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=DEFAULT_CHAT_MODEL,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=1,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        learning_rate=5e-5,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        report_to="none",  # disable wandb etc. unless configured
    )

    # Collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save model + tokenizer
    model.save_pretrained(DEFAULT_CHAT_MODEL)
    tokenizer.save_pretrained(DEFAULT_CHAT_MODEL)
    print(f"✅ Model saved to {DEFAULT_CHAT_MODEL}")


if __name__ == "__main__":
    main()
