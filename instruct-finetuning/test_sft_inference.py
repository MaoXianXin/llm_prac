from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/mao/workspace/llm_prac/instruct-finetuning/model_16bit",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    dtype=None,
)

tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

model = FastLanguageModel.for_inference(model)

messages = [
    {"from": "human", "value": "Is 9.11 larger than 9.9?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

print(tokenizer.eos_token)
print(tokenizer.pad_token)
print(tokenizer.bos_token)
text_streamer = TextStreamer(tokenizer)
_ = model.generate(
    input_ids=inputs,
    streamer=text_streamer, 
    max_new_tokens=128, 
    use_cache=False
)