from unsloth import FastLanguageModel
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.bfloat16 # Match the dtype used in training
load_in_4bit = True # Use 4bit quantization to reduce memory usage

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit", # Your saved LoRA model from training
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

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

# Load the Korean Wikipedia dataset (same as in SFT_QLoRA_Qwen2.5-0.5B.py)
print("Loading Korean Wikipedia dataset...")
wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train[:1%]")  # Load the first 1% of the training set

# Apply the same template as in SFT_QLoRA_Qwen2.5-0.5B.py
def apply_template(examples):
    # Prepare messages in ChatML format for each example
    all_messages = []
    for i in range(len(examples["title"])):
        # Use title for the user prompt and text for the assistant response
        messages = [
            {"from": "human", "value": f"{examples['title'][i]}"},
            {"from": "gpt", "value": examples['text'][i]}
        ]
        all_messages.append(messages)

    # Apply the chat template to the messages
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in all_messages]
    return {"text": text}

# Format the dataset with the template
print("Formatting dataset with chat template...")
test_dataset = wiki_dataset.map(apply_template, batched=True, remove_columns=list(wiki_dataset.features))
test_texts = test_dataset["text"]

# Calculate perplexity
print(f"Evaluating model on {len(test_texts)} Wikipedia samples...")
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
    f.write(f"Test Dataset: Korean Wikipedia (20231101.ko, 1% sample)\n")
    f.write(f"Number of test samples: {len(test_texts)}\n")
    f.write(f"Average Perplexity: {avg_perplexity:.4f}\n")
    f.write(f"Median Perplexity: {median_perplexity:.4f}\n")
    f.write(f"Min Perplexity: {min_perplexity:.4f}\n")
    f.write(f"Max Perplexity: {max_perplexity:.4f}\n")