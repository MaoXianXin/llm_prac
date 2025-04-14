from unsloth import FastLanguageModel
import torch
import os
import glob
import numpy as np
from tqdm import tqdm
from datasets import Dataset

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.bfloat16 # Match the dtype used in training
load_in_4bit = True # Use 4bit quantization to reduce memory usage

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # Your saved LoRA model from training
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# Ensure pad_token is set to eos_token as in training
tokenizer.pad_token = tokenizer.eos_token

# Function to load test data
def load_test_files(directory_path):
    """Load all txt files from the specified directory"""
    all_texts = []
    for txt_file in glob.glob(os.path.join(directory_path, "*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:  # Ensure text is not empty
                all_texts.append(text)
    
    return {"text": all_texts}

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, texts, max_length=max_seq_length):
    """Calculate perplexity for a list of texts"""
    model.eval()
    perplexities = []
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating perplexity"):
            # Tokenize input
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to("cuda")
            attention_mask = encodings.attention_mask.to("cuda")
            
            # Get model output
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Calculate perplexity
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
    
    return perplexities

# Path to your test dataset
test_data_path = "/home/mao/workspace/medium_scrape/chunks"  # Replace with your actual test data path

# Load test data
test_dataset = load_test_files(test_data_path)
test_texts = test_dataset["text"]

# Calculate perplexity
print(f"Evaluating model on {len(test_texts)} test samples...")
perplexities = calculate_perplexity(model, tokenizer, test_texts)

# Print results
avg_perplexity = np.mean(perplexities)
min_perplexity = np.min(perplexities)
max_perplexity = np.max(perplexities)
median_perplexity = np.median(perplexities)

print(f"\nPerplexity Evaluation Results:")
print(f"Average Perplexity: {avg_perplexity:.4f}")
print(f"Median Perplexity: {median_perplexity:.4f}")
print(f"Min Perplexity: {min_perplexity:.4f}")
print(f"Max Perplexity: {max_perplexity:.4f}")

# Optional: Save results to file
with open("perplexity_results.txt", "w") as f:
    f.write(f"Test Dataset: {test_data_path}\n")
    f.write(f"Number of test samples: {len(test_texts)}\n")
    f.write(f"Average Perplexity: {avg_perplexity:.4f}\n")
    f.write(f"Median Perplexity: {median_perplexity:.4f}\n")
    f.write(f"Min Perplexity: {min_perplexity:.4f}\n")
    f.write(f"Max Perplexity: {max_perplexity:.4f}\n")

# Simple test prompt (original code)
test_prompt = "标题：The $5 SaaS Revolution:"
inputs = tokenizer(
    [test_prompt], 
    return_tensors = "pt"
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)