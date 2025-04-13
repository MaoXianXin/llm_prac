from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen2.5-1.5B-FP16",
    messages=[
        {"role": "user", "content": "Give three tips for staying healthy."},
        {"role": "user_context", "content": "You are a doctor."},
    ],
    max_tokens=500
)
print("Chat response:", chat_response)

# Extract the content from the response
content = chat_response.choices[0].message.content

print("Chat response content:", content)