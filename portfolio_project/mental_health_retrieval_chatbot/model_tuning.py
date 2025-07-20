import os
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

from .constants import (
    DEFAULT_BASE_CHAT_MODEL,
    DEFAULT_CHAT_MODEL,
    EMPATHETIC_DIALOGUES_PATH,
    DAILY_DIALOG_PATH,
    Names,
)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_CHAT_MODEL)

# Set pad_token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(DEFAULT_BASE_CHAT_MODEL)
model.resize_token_embeddings(len(tokenizer))  # Important if you add special tokens


# 1. Load EmpatheticDialogues
def load_empathetic(path: str = EMPATHETIC_DIALOGUES_PATH):
    df = pd.read_csv(path, quoting=1, on_bad_lines="skip")
    df[Names.TEXT] = df["prompt"] + " " + df["utterance"]
    return Dataset.from_pandas(df[[Names.TEXT]])


# 2. Load DailyDialog
def load_dailydialog(path: str = DAILY_DIALOG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    conversations = [line.strip().replace("__eou__", "") for line in lines]
    df = pd.DataFrame(conversations, columns=[Names.TEXT])
    return Dataset.from_pandas(df)


# 3. Preprocessing
def tokenize_function(example: pd.DataFrame) -> AutoTokenizer:
    return tokenizer(example[Names.TEXT], truncation=True, max_length=512)


def main():
    # Load datasets
    empathic_ds = load_empathetic()
    daily_ds = load_dailydialog()

    # Combine and shuffle
    combined_ds = concatenate_datasets([empathic_ds, daily_ds]).shuffle(seed=42)

    # Tokenize
    tokenized = combined_ds.map(
        tokenize_function, batched=True, remove_columns=[Names.TEXT]
    )

    # Define training args
    training_args = TrainingArguments(
        output_dir=DEFAULT_CHAT_MODEL,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        learning_rate=0.00005,
        weight_decay=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1,
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Fine-tune!
    trainer.train()

    # Save model
    model.save_pretrained(DEFAULT_CHAT_MODEL)
    tokenizer.save_pretrained(DEFAULT_CHAT_MODEL)


if __name__ == "__main__":
    main()
