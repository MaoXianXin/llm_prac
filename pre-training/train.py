from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/mao/workspace/llm_prac/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323",
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
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

# 新的数据加载和处理函数
def load_text_files(directory_path):
    """加载指定目录下的所有txt文件"""
    import os
    import glob
    
    all_texts = []
    for txt_file in glob.glob(os.path.join(directory_path, "*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:  # 确保文本不为空
                all_texts.append(text)
    
    return {"text": all_texts}

# 加载文本文件
articles_path = "/home/mao/workspace/medium_scrape/articles"  # 你的文章目录
dataset = load_text_files(articles_path)

# 将数据转换为HuggingFace Dataset格式
from datasets import Dataset
dataset = Dataset.from_dict(dataset)

# 对文本进行分词处理
def tokenize_function(examples):
    # 先进行tokenization，不添加特殊标记
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=max_seq_length-1,  # 预留一个位置给EOS标记
        add_special_tokens=False  # 不自动添加特殊标记
    )
    
    # 手动在每个序列末尾添加EOS标记
    for i in range(len(tokenized_inputs["input_ids"])):
        tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
        if "attention_mask" in tokenized_inputs:
            tokenized_inputs["attention_mask"][i].append(1)
    
    return tokenized_inputs

# 对数据集进行预处理
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=2,
    remove_columns=["text"]
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    # 不再需要dataset_text_field参数，因为我们已经预处理了数据
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,  # 对于预训练任务，启用packing可以提高效率
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # num_train_epochs=1,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
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



# 测试生成
FastLanguageModel.for_inference(model)  # 启用更快的推理

# 使用更简单的提示进行测试
test_prompt = "继续下面的文本：人工智能的发展历程"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)


model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")


model.save_pretrained_merged("model_16bit", tokenizer, save_method = "merged_16bit",)
model.save_pretrained_merged("model_4bit", tokenizer, save_method = "merged_4bit_forced",)