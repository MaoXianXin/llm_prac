from unsloth import FastLanguageModel

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

print(tokenizer.eos_token)
print(tokenizer.pad_token)
print(tokenizer.bos_token)

# Wikipedia provides a title and an article text.
# Use https://translate.google.com!
_wikipedia_prompt = """Wikipedia Article
### Title: {}

### Article:
{}"""
# becomes:
wikipedia_prompt = """위키피디아 기사
### 제목: {}

### 기사:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    titles = examples["title"]
    texts  = examples["text"]
    outputs = []
    for title, text in zip(titles, texts):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = wikipedia_prompt.format(title, text) + EOS_TOKEN
        outputs.append(text)
    return { "text" : outputs, }

from datasets import load_dataset

dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split = "train",)

# We select 1% of the data to make training faster!
dataset = dataset.train_test_split(train_size = 0.01)["train"]
# 打印原始数据集的一个样本
print("原始数据集样本:")
print(dataset[0])

dataset = dataset.map(formatting_prompts_func, batched = True,)
# 打印处理后的数据集的一个样本
print("\n处理后的数据集样本:")
print(dataset[0])

print(tokenizer.eos_token)
print(tokenizer.pad_token)
print(tokenizer.bos_token)