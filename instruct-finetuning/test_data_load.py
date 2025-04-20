from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

print(tokenizer.eos_token)
print(tokenizer.pad_token)
print(tokenizer.bos_token)


tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
# 打印原始数据集的一个样本
print("原始数据集样本:")
print(dataset[0])
dataset = dataset.map(apply_template, batched=True)
# 打印处理后的数据集的一个样本
print("\n处理后的数据集样本:")
print(dataset[0])

print(tokenizer.eos_token)
print(tokenizer.pad_token)
print(tokenizer.bos_token)

