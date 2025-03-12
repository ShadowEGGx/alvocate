import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import os

# Load the cleaned datasets
aptitude_data = pd.read_csv("data/processed_aptitude.csv")
logical_reasoning_data = pd.read_csv("data/processed_logical_reasoning.csv")
cse_data = pd.read_csv("data/processed_cse.csv")
leetcode_data = pd.read_csv("data/processed_leetcode.csv")

# Combine all datasets into one
all_data = pd.concat([aptitude_data, logical_reasoning_data, cse_data, leetcode_data], ignore_index=True)

# Convert DataFrame to HuggingFace Dataset
dataset = Dataset.from_pandas(all_data)

# Load pre-trained transformer model and tokenizer (GPT-like model)
model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # Example: You can replace with "gpt-3.5-turbo" or "Llama-2"
token = os.getenv("HUGGINGFACE_TOKEN")
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["question"] + " " + examples["answer"], truncation=True, padding="max_length")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./trained_model",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

# Data collator for masked language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set to True if training for masked language modeling (MLM)
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,  # Can be split into train-test if needed
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save trained model and tokenizer
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

print("Training complete! Model saved in './trained_model'")
