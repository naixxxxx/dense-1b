from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base",use_fast=True,)

EOD_ID = tokenizer.convert_tokens_to_ids("<|endoftext|>")

text = "Hello world!"

ids = tokenizer.encode(text,add_special_tokens=False)
ids.append(EOD_ID)
print(ids)