# %%

from langchain_community.document_loaders import PyMuPDFLoader

file_path = "/home/mao/Downloads/《置身时代的社会理论》史蒂文·塞德曼【文字版_PDF电子书_雅书】.pdf"
loader = PyMuPDFLoader(file_path, mode="single")

#%%

docs = loader.load()
print(len(docs))

#%%

pdf_text = docs[0].page_content.replace('\n', '')
# %%

def count_tokens(text, tokenizer):
    # Tokenize the text
    tokens = tokenizer.encode(text)
    
    # Return the token count
    return len(tokens)

#%%

from transformers import AutoTokenizer

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/mao/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987")

# 使用函数计算tokens
token_count = count_tokens(text=pdf_text, tokenizer=tokenizer)
print(f"Token count: {token_count}")
# %%
