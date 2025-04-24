import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import torch
import jieba # <-- Added for Chinese tokenization
from tqdm import tqdm # <-- Added for progress bar

# --- Unsloth/Transformers Imports (from test_sft_inference.py) ---
from unsloth import FastLanguageModel
# from transformers import TextStreamer # Not needed for batch processing
from unsloth.chat_templates import get_chat_template

# --- Configuration ---
json_file_path = './translation_data.json'
model_path = "/home/mao/workspace/llm_prac/instruct-finetuning/output/checkpoint-471" # <-- Specify your model path
max_seq_length = 2048
load_in_4bit = True # Set to False if you have enough VRAM and want faster inference
num_samples_to_evaluate = 100 # <-- How many samples to evaluate? Set to None to evaluate all.

# --- 1. 加载翻译数据 ---
print(f"正在从文件读取数据: {json_file_path}")
try:
    with open(json_file_path, 'r', encoding='utf-8') as f_json:
        translation_data = json.load(f_json)
    print(f"成功加载 {len(translation_data)} 条翻译数据。")
    if num_samples_to_evaluate is not None:
        translation_data = translation_data[:num_samples_to_evaluate]
        print(f"将评估前 {num_samples_to_evaluate} 条数据。")
except FileNotFoundError:
    print(f"错误：文件未找到 '{json_file_path}'。")
    exit()
except json.JSONDecodeError:
    print(f"错误：无法解析 JSON 文件 '{json_file_path}'。")
    exit()
except Exception as e:
    print(f"读取文件时发生错误: {e}")
    exit()

# --- 2. 加载LLM模型和分词器 ---
print(f"正在加载模型: {model_path}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
    dtype=None, # dtype is automatically handled by Unsloth
)

# --- 3. 设置Chat Template ---
# Make sure the mapping matches how your model was trained
tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"}, # Adjust if your roles are different
    chat_template="chatml", # Or the template your model expects
)
# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("设置 pad_token 为 eos_token")

# --- 4. 准备模型进行推理 ---
model = FastLanguageModel.for_inference(model)
print("模型已准备好进行推理。")

# --- 5. 评估循环 ---
bleu_scores = []
chencherry = SmoothingFunction()

print(f"\n开始评估 {len(translation_data)} 个样本...")
for item in tqdm(translation_data):
    en_sentence = item.get('en')
    ref_translation = item.get('ch')

    if not en_sentence or not ref_translation:
        print(f"跳过不完整的条目: {item}")
        continue

    # --- 准备模型输入 ---
    # We add a clear instruction for the model
    messages = [
        {"from": "human", "value": f"{en_sentence}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True, # Add the prompt for the assistant's turn
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu") # Move inputs to GPU if available

    # --- 生成翻译 (Hypothesis) ---
    # Note: We don't use TextStreamer here as we need the final output string
    # Adjust max_new_tokens based on expected translation length
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=128, # Adjust as needed
            use_cache=True, # Enable cache for faster generation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens, skipping the input prompt
    # outputs[0] takes the first (and only) batch element
    # inputs.shape[1] gives the length of the input sequence
    generated_ids = outputs[0, inputs.shape[1]:]
    hyp_translation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # --- 预处理 (Tokenization using Jieba) ---
    # Reference needs to be a list of lists of tokens
    reference_tokens = [list(jieba.cut(ref_translation))]
    # Hypothesis needs to be a list of tokens
    hypothesis_tokens = list(jieba.cut(hyp_translation))

    # --- 计算 BLEU 分数 ---
    try:
        bleu_score = sentence_bleu(
            reference_tokens,
            hypothesis_tokens,
            smoothing_function=chencherry.method4 # Common smoothing method
        )
        bleu_scores.append(bleu_score)
    except ZeroDivisionError:
        print(f"警告: Hypothesis 为空，无法计算 BLEU。")
        print(f"  EN: {en_sentence}")
        print(f"  REF: {ref_translation}")
        print(f"  HYP: {hyp_translation}")
        bleu_scores.append(0.0) # Assign 0 score for empty hypothesis
    except Exception as e:
        print(f"计算 BLEU 时发生错误: {e}")
        print(f"  EN: {en_sentence}")
        print(f"  REF: {ref_translation}")
        print(f"  HYP: {hyp_translation}")
        bleu_scores.append(0.0) # Assign 0 score on error

    # Optional: Print individual results for debugging
    # print(f"\nEN: {en_sentence}")
    # print(f"REF: {ref_translation}")
    # print(f"HYP: {hyp_translation}")
    # print(f"BLEU: {bleu_score:.4f}")

# --- 6. 计算并输出平均 BLEU 分数 ---
if bleu_scores:
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"\n--- 评估完成 ---")
    print(f"评估样本数: {len(bleu_scores)}")
    print(f"平均 BLEU 分数: {average_bleu:.4f}")

    # --- Interpretation of Average Score ---
    if average_bleu > 0.5:
        print("整体解释: 翻译质量较高。")
    elif average_bleu > 0.3:
        print("整体解释: 翻译质量中等，可能存在一些不流畅或不准确之处。")
    elif average_bleu > 0.1:
        print("整体解释: 翻译质量较低，可能难以理解。")
    else:
        print("整体解释: 翻译质量非常低。")
else:
    print("\n没有计算任何 BLEU 分数。请检查数据和配置。")
