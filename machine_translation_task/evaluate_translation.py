# evaluate_translation.py
import json
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer # Keep if you want streaming for debugging, remove for pure evaluation
from tqdm import tqdm # For progress bar
import nltk # For BLEU score calculation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

# --- 配置 ---
json_file_path = './translation_data.json' # 确保路径正确
model_name = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
max_seq_length = 2048
dtype = None
load_in_4bit = True
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens_gen = 128 # Max tokens to generate for translation

# --- 1. 加载数据 ---
print(f"正在从文件读取数据: {json_file_path}")
try:
    with open(json_file_path, 'r', encoding='utf-8') as f_json:
        translation_data = json.load(f_json)
    print(f"成功加载 {len(translation_data)} 条翻译数据。")
except FileNotFoundError:
    print(f"错误：文件未找到 '{json_file_path}'。")
    exit()
except json.JSONDecodeError:
    print(f"错误：无法解析 JSON 文件 '{json_file_path}'。")
    exit()
except Exception as e:
    print(f"读取文件时发生错误: {e}")
    exit()

# --- 2. 加载模型和 Tokenizer ---
print(f"正在加载模型: {model_name}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
print("模型加载完成。")

# --- 3. 定义 Prompt 模板 ---
# 使用更明确的翻译指令
chatml_prompt_template = """<|im_start|>system
You are a helpful assistant specialized in translation. Translate the following English sentence into Chinese.<|im_end|>
<|im_start|>user
English: {}
Chinese:<|im_end|>
<|im_start|>assistant
"""

# --- 4. & 5. 推理和评估 ---
references = []
hypotheses = []
chencherry = SmoothingFunction() # BLEU smoothing

print("开始进行翻译评估...")
# 使用 tqdm 显示进度条

# Limit the data to the first 100 items (or fewer if the dataset is smaller)
num_samples_to_evaluate = 100
data_to_evaluate = translation_data[:num_samples_to_evaluate]
print(f"将仅评估前 {len(data_to_evaluate)} 条数据。") # Inform the user

# Update the loop to iterate over the sliced data
for item in tqdm(data_to_evaluate, desc=f"Evaluating first {len(data_to_evaluate)} samples"):
    en_text = item.get('en')
    ref_ch_text = item.get('ch')

    if not en_text or not ref_ch_text:
        print(f"跳过无效数据项: {item}")
        continue

    # 构建 Prompt
    prompt = chatml_prompt_template.format(en_text)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    # 模型生成 (不使用 streamer 获取最终输出)
    # eos_token_id for Qwen2.5-0.5B-Instruct is 151645 (<|im_end|>)
    # You might need to adjust eos_token_id based on the specific model version if needed.
    # Using tokenizer.eos_token_id is generally safer if defined.
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 151645

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens_gen,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id, # Avoid warning
        do_sample=False # Use greedy decoding for deterministic output
    )

    # 解码生成的文本，跳过 prompt 部分和特殊 token
    # inputs.input_ids.shape[1] gives the length of the prompt tokens
    output_tokens = outputs[0, inputs.input_ids.shape[1]:]
    hyp_ch_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    # 准备 BLEU 计算所需格式 (这里使用字符级 BLEU 作为简单示例)
    # NLTK's corpus_bleu expects: list of references, list of hypotheses
    # Each reference in the list of references should be a list of sentences (if multiple refs per hyp)
    # Each sentence should be a list of tokens.
    # Here: one reference per hypothesis, tokenizing by character.
    reference_tokens = [list(ref_ch_text)] # List containing one reference sentence, tokenized by char
    hypothesis_tokens = list(hyp_ch_text)   # Hypothesis sentence, tokenized by char

    references.append(reference_tokens)
    hypotheses.append(hypothesis_tokens)

    # (可选) 打印一些样本进行检查
    if len(hypotheses) % 50 == 0: # Print every 50 samples
       print(f"\n--- Sample {len(hypotheses)} ---")
       print(f"  EN: {en_text}")
       print(f"  REF CH: {ref_ch_text}")
       print(f"  HYP CH: {hyp_ch_text}")
       # Calculate sentence BLEU for this sample (optional)
       # sample_bleu = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=chencherry)
       # print(f"  Sample BLEU: {sample_bleu:.4f}")

# --- 6. 计算整体 BLEU 分数 ---
print("\n计算整体 BLEU 分数...")
if references and hypotheses:
    # weights can be adjusted, (1.0,) for BLEU-1, (0.5, 0.5) for BLEU-2, etc.
    # Default is BLEU-4: (0.25, 0.25, 0.25, 0.25)
    corpus_bleu_score = corpus_bleu(references, hypotheses, smoothing_function=chencherry.method7)
    print(f"\n评估完成!")
    print(f"数据集: {json_file_path}")
    print(f"模型: {model_name}")
    print(f"评估样本数: {len(hypotheses)}") # Updated label for clarity
    print(f"整体 Corpus BLEU-4 分数 (字符级): {corpus_bleu_score:.4f}")
    print(f"BLEU 分数越高越好 (范围 0 到 1)。")
else:
    print("没有有效的翻译结果可供评估。")

# --- (可选) 安装 NLTK ---
# 如果你还没有安装 nltk，请运行: pip install nltk tqdm
# 第一次使用 nltk 可能需要下载数据:
# import nltk
# nltk.download('punkt') # punkt 是常用的分词器数据，虽然这里字符级BLEU不直接用，但安装nltk时建议下载