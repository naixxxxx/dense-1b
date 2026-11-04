import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

path = "raw_data/part-00000.parquet"

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B-Base",
    use_fast=True,
)

EOD_ID = tokenizer.convert_tokens_to_ids("<|endoftext|>")

parquet_file = pq.ParquetFile(path)                                                                                                                                

with open("data/train_data.bin", "ab") as f:

    for batch in parquet_file.iter_batches(
        batch_size=1024,
        columns=["content"],
    ):
        texts = batch.column("content").to_pylist() # texts is a list of strings

        encoded_list = tokenizer(  
            texts, 
            add_special_tokens=False,
        )["input_ids"] # list[list[int]]

        id_list = []

        for ids in encoded_list:
            id_list.extend(ids)
            id_list.append(EOD_ID)

        np.asarray(id_list, dtype=np.uint32).tofile(f)
        