from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/mao/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

tokenizer.pad_token = "<|endoftext|>"

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

EOS_TOKEN = "<|im_end|>" # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    # 修改函数以适应实际数据格式
    conversations_list = examples.get("conversations", [])
    texts = []
    
    for conversations in conversations_list:
        formatted_text = ""
        for message in conversations:
            # 将 "from" 映射到 "role"，将 "value" 映射到 "content"
            if message['from'] == "system":
                role = "system"
            elif message['from'] == "user":
                role = "user"
            else: # 默认视为 assistant
                role = "assistant"
            formatted_text += f"<|im_start|>{role}\n{message['value']}<|im_end|>\n"
        
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
        # num_train_epochs = 5, # Set this for 1 full training run.
        max_steps = 100,
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
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
标题：Fine-Tuning the Qwen2-VL Model: A Comprehensive Guide\n\nURL：https://moazharu.medium.com/fine-tuning-the-qwen2-vl-model-a-comprehensive-guide-75e86cdcfc2d\n\n正文：\nFine-Tuning the Qwen2-VL Model: A Comprehensive Guide\nazhar\n·\nFollow\nPublished in\nazhar labs\n·\n5 min read\n·\nSep 12, 2024\n--\n1\nThe growing landscape of AI and machine learning has seen significant advancements, especially in the domain of multimodal models — those that can process and analyze multiple types of data (text, images, videos, etc.) simultaneously. A notable example of such technology is the Qwen2-VL model developed by Alibaba Cloud. In this article, we will explore how to fine-tune this model using the Llama Factory framework, a low-code environment designed for efficient training and usage of large language models (LLMs).\nBefore we proceed, let’s stay connected! Please consider following me on Medium, and don’t forget to connect with me on LinkedIn for a regular dose of data science and deep learning insights.” 🚀📊🤖\nUnderstanding Multimodal Models\nMultimodal models are essential for numerous applications today, including image captioning, video analysis, and even interactive AI applications such as chatbots that comprehend both visual and textual inputs. Qwen2-VL encapsulates these capabilities, and its design reflects an emphasis on accessibility, reducing the barriers for developers and researchers by providing an open-source and relatively lightweight solution.\nWhy Fine-Tune?\nFine-tuning is a process whereby a pretrained model is further trained (or tuned) on a specific dataset. This allows the model to adapt to tasks that might not have been included in its initial training phases. Fine-tuned models tend to perform better on domain-specific tasks and variables since they have learned to adjust their internal parameters to fit the nuances of the new data. This makes fine-tuning critical for anyone looking to leverage AI for targeted results.\nPreparing to Fine-Tune with Llama Factory\nWhat is Llama Factory?\nLlama Factory is a framework designed to facilitate the fine-tuning of LLMs, incorporating both low-code and no-code approaches. It allows users to manage model training efficiently without needing thorough programming expertise, making it a fit for a wide array of practitioners in the AI field.\nSetup Requirements\nTo get started, you will need:\nA GPU with at least 12GB of RAM.\nPython (3.7 or above) and required libraries.\nAccess to a code execution environment (such as Google Colab, RunPod, Lambda Labs, or a local setup with a powerful GPU).\nClone the Llama Factory repository\ngit clone https://github.com/hiyouga/LLaMA-Factory.git\nNavigate into the cloned directory\ncd LLaMA-Factory\nInstall required packages\n!pip install -r requirements.txt \n!pip install bitsandbytes \n!pip install git+https://github.com/huggingface/transformers.git \n!pip install -e \".[torch, metrics]\" \n!pip install liger-kernel\nRestart your runtime if you’re using Google Colab or Jupyter to reset the environment\nInitializing the Environment\nOnce the environment is set up and the packages installed, run the commands to set up the Llama Factory CLI or Llama Board. We will use the command line interface (CLI).\nimport os \n!GRADIO_SHARE=1 llamafactory-cli webui\nPreparing the Data\nBefore diving into fine-tuning, it is important to prepare the datasets that will be used. You can utilize a sample dataset like mllm_demo.\nData Selection and Structure\nIn Llama Factory, data is expected in a specific format, typically structured as JSON files. You can also create a new dataset or alter existing datasets for your specific needs. Ensure the data is preprocessed effectively to avoid errors during the training phase.\nConfiguration Details\nConfigurations in Llama Factory can be set through YAML files or directly through the CLI. Create a JSON configuration file with properties relevant to your fine-tuning needs.\nHere’s an example:\n{ \"stage\": \"sft\", \n\"do_train\": true, \n\"model_name_or_path\": \"Qwen/Qwen2-VL-2B-Instruct\", \n\"dataset\": \"mllm_demo,identity\", \n\"template\": \"qwen2_vl\", \n\"finetuning_type\": \"lora\", \n\"lora_target\": \"all\", \n\"output_dir\": \"qwen2vl_lora\", \n\"per_device_train_batch_size\": 2, \n\"gradient_accumulation_steps\": 4, \n\"lr_scheduler_type\": \"cosine\", \n\"logging_steps\": 10, \n\"warmup_ratio\": 0.1, \n\"save_steps\": 1000, \n\"learning_rate\": 5e-5, \n\"num_train_epochs\": 3.0, \n\"max_samples\": 500, \n\"max_grad_norm\": 1.0, \n\"loraplus_lr_ratio\": 16.0, \n\"fp16\": true, \n\"use_liger_kernel\": true }\nThis JSON structure outlines parameters essential for fine-tuning, such as model path, dataset selection, output directory, and several hyperparameters that govern the training process.\nExecuting Fine-Tuning\nUsing CLI\nCreate the Fine-tuning Script\nTo initiate the fine-tuning process, the JSON configuration file we created can be used as follows:\nimport json\n\nargs = { \"model_name_or_path\": \"Qwen/Qwen2-VL-2B-Instruct\", \"do_train\": True, \"dataset\": \"mllm_demo,identity\", \"template\": \"qwen2_vl\", \"finetuning_type\": \"lora\", \"lora_target\": \"all\", \"output_dir\": \"qwen2vl_lora\", \"per_device_train_batch_size\": 2, \"gradient_accumulation_steps\": 4, \"learning_rate\": 5e-5, \"num_train_epochs\": 3 }\nSave to a JSON file\nwith open(\"train_qwen2vl.json\", \"w\", encoding=\"utf-8\") as f: \n    json.dump(args, f, ensure_ascii=False, indent=4)\nStart the training process\n!llamafactory-cli train train_qwen2vl.json\nMonitoring the Fine-tuning Process\nDuring the fine-tuning process, Llama Factory will output logs detailing the training progress. You can monitor the logging_steps parameter to see how often the logs appear. Adjust this value based on the verbosity you desire.\nExpected Outcomes\nUpon successful fine-tuning, you should see logs that indicate performance metrics such as loss reduction and accuracy of the model on the training dataset.\nMerging the Fine-Tuned Model\nOnce the model has been successfully fine-tuned, you may want to merge LoRA (Low-Rank Adaptation) adapters into your model. This step is crucial as it helps in making efficient use of learned parameters without excessively increasing the model size.\nTutorial on Merging Adapters\nHere’s an example of how the merging can be conducted:\nargs = { \"model_name_or_path\": \"Qwen/Qwen2-VL-2B-Instruct\", \"adapter_name_or_path\": \"qwen2vl_lora\", \"template\": \"qwen2_vl\", \"finetuning_type\": \"lora\", \"export_dir\": \"qwen2vl_2b_instruct_lora_merged\", \"export_size\": 2, \"export_device\": \"cpu\" }\nSave to a JSON file\nwith open(\"merge_qwen2vl.json\", \"w\", encoding=\"utf-8\") as f: \n    json.dump(args, f, ensure_ascii=False, indent=4)\nStart the merge process\n!llamafactory-cli export merge_qwen2vl.json\nPushing to Hugging Face\nYou can upload your merged model to Hugging Face’s model hub for easy access and sharing. Here’s how you can do it:\nfrom huggingface_hub import notebook_login\n\nnotebook_login()\n\nfrom huggingface_hub import HfApi\nCreate an instance of HfApi\napi = HfApi() \nfinal_model_path = \"/content/LLaMA-Factory/qwen2vl_2b_instruct_lora_merged\" \nhf_model_repo = \"your_username/Qwen2-VL-2B-Instruct-LoRA-FT\"\nUpload the merged model\napi.upload_folder(folder_path=final_model_path, repo_id=hf_model_repo, commit_message=\"Initial model upload\")\n\nprint(f\"Model pushed to: {hf_model_repo}\") \nConclusion\nThe Qwen2-VL model at the cutting edge of multimodal AI is becoming increasingly popular due to its capabilities and ease of use. Llama Factory provides an excellent framework for those interested in fine-tuning such advanced models, making the process accessible to a wider audience.\nFine-tuning, merging, and sharing LLMs has now become more practical and intuitive than ever. With the appropriate setup and understanding of the underlying tools, you can unlock the potential of these models in unique and powerful ways.\nNext Steps\nExperiment: Try fine-tuning the model with different datasets to see how performance varies based on data quality and type.\nExplore: Check out advanced configurations in Llama Factory to optimize the training procedure for speed and efficiency.\nEngage: Join communities that focus on AI and machine learning to share insights and collaborate on projects using Qwen2-VL.\nFor feedback, discussions, or community support, consider joining Discord servers dedicated to AI, where you can learn from others’ experiences. The AI landscape is vast, and continuous learning is key to maximizing its potential.<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer([chatml_input], return_tensors="pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_special_tokens=False, skip_prompt=True)
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=2048,
    eos_token_id=[151645, 151643]
)


model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")


# model.save_pretrained_merged("model_16bit", tokenizer, save_method = "merged_16bit",)
# model.save_pretrained_merged("model_4bit", tokenizer, save_method = "merged_4bit_forced",)