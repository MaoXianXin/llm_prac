from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/mao/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987",
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=True,
)

# 明确设置 pad_token 为 eos_token
tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# Replace the alpaca_prompt with ChatML template
# We'll use the template from the template_chatml.jinja file
chatml_template = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    # 修改函数以适应实际数据格式
    conversations_list = examples.get("conversations", [])
    texts = []
    
    for conversations in conversations_list:
        formatted_text = ""
        for message in conversations:
            # 将 "from" 映射到 "role"，将 "value" 映射到 "content"
            role = "user" if message['from'] == "user" else "assistant"
            formatted_text += f"<|im_start|>{role}\n{message['value']}<|im_end|>\n"
        
        # 如果最后一条消息不是助手，添加助手提示
        if conversations and conversations[-1]['from'] != "assistant":
            formatted_text += "<|im_start|>assistant\n"
        
        # 添加 EOS_TOKEN
        formatted_text += EOS_TOKEN
        texts.append(formatted_text)
    
    return {"text": texts}

from datasets import load_dataset
# 更新数据集路径
dataset = load_dataset(
    "json", 
    data_files="/home/mao/workspace/medium_scrape/flask_endpoint/data_preparation/conversation_records.json",  # 请替换为您的实际数据集路径
    split="train"
)

# 打印原始数据集的第一条数据
print("原始数据集的第一条数据:")
print(dataset[0])

dataset = dataset.map(formatting_prompts_func, batched=True,)

# 打印处理后数据集的第一条数据
print("\n处理后数据集的第一条数据:")
print(dataset[0])

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 20,
        num_train_epochs = 5, # Set this for 1 full training run.
        # max_steps = 100,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")



FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# Example of a ChatML formatted input for inference
chatml_input = """<|im_start|>system
You are a helpful AI assistant.
<|im_end|>
<|im_start|>user
Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8
<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer([chatml_input], return_tensors="pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)


# model.save_pretrained("lora_model")  # Local saving
# tokenizer.save_pretrained("lora_model")


# model.save_pretrained_merged("model_16bit", tokenizer, save_method = "merged_16bit",)
# model.save_pretrained_merged("model_4bit", tokenizer, save_method = "merged_4bit_forced",)