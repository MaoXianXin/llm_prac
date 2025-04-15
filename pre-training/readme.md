
# 训练准备(train.py)

max_seq_length = 2048: 设置模型能处理的最大序列长度，在训练不同阶段可以逐步增加，优化模型对长文本的处理能力

首先需要选择是LoRA微调还是全量参数微调
LoRA微调相关参数设置:
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
一般来讲是4bit为True
r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
这个参数影响可微调的参数量，越大可微调的参数量越大，训练时长越长
可以开启model.save_pretrained()、model.save_pretrained_merged()这两个选择

全量参数微调相关参数设置:
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=True,
需要把model.save_pretrained()、model.save_pretrained_merged()这两个地方注释掉

model_name: 设置待加载的预训练模型位置

接下来设置数据路径:
articles_path = "/home/mao/workspace/medium_scrape/chunks"

模型训练相关参数设置:
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=10,
        # max_steps=100,
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

其中比较关键的是:
num_train_epochs: 涉及训练时长，也会影响模型拟合程度
其余的per_device_train_batch_size、warmup_steps、learning_rate、lr_scheduler_type根据情况修改

# 推理准备(inference.py)

max_seq_length = 2048: 需要保持跟训练时一致

如果是bfloat16微调出来的模型，可以设置load_in_4bit = False
model_name: 设置为待加载的模型路径

设置测试数据集路径:
test_data_path = "/home/mao/workspace/medium_scrape/chunks"