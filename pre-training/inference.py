from unsloth import FastLanguageModel
import torch
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

# Simple test prompt similar to what you used in training
test_prompt = "标题：The $5 SaaS Revolution:"
inputs = tokenizer(
    [test_prompt], 
    return_tensors = "pt"
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)