from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 使用更适合中文预训练模型的提示
chat_response = client.chat.completions.create(
    model="Qwen2.5-1.5B-FP16",  # 确保这里的模型名称与你在vLLM中部署的模型名称一致
    messages=[
        {"role": "user", "content": "标题：1-bit Quantization: Run Models with Trillions of Parameters on"},
    ],
    max_tokens=500,
    temperature=0.1,  # 可以调整生成的随机性
    stream=False  # 设置为True可以获得流式响应
)
print("Chat response:", chat_response)

# Extract the content from the response
content = chat_response.choices[0].message.content

print("Chat response content:", content)