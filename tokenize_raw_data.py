from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

text = "你好，世界！"

ids = tokenizer.encode(text)

print(ids)
print(tokenizer.decode(ids))