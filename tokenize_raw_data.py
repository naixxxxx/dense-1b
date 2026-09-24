from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


RAW_DATA_DIR = Path("data/raw_data")
OUTPUT_PATH = Path("data/en_qa.bin")

BATCH_SIZE = 1024


tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B-Base",
    use_fast=True,
)

eos_id = tokenizer.eos_token_id

print(f"EOS token: {tokenizer.eos_token}")
print(f"EOS token id: {eos_id}")


parquet_files = sorted(RAW_DATA_DIR.glob("*.parquet"))

if not parquet_files:
    raise RuntimeError(f"No parquet files found in {RAW_DATA_DIR}")


total_tokens = 0
total_documents = 0


with open(OUTPUT_PATH, "wb") as fout:

    for parquet_path in parquet_files:
        print(f"\nProcessing: {parquet_path}")

        parquet = pq.ParquetFile(parquet_path)

        batches = parquet.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["content"],
        )

        for batch in tqdm(
            batches,
            total=(parquet.metadata.num_rows + BATCH_SIZE - 1) // BATCH_SIZE,
        ):
            texts = batch.column("content").to_pylist()

            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]

            # 每个 document:
            #
            # token token token ... EOS
            #
            # 然后所有 document 连成一条长序列。
            for ids in encoded:
                ids.append(eos_id)

                np.asarray(
                    ids,
                    dtype=np.uint32,
                ).tofile(fout)

                total_tokens += len(ids)
                total_documents += 1


print("\nDone.")
print(f"Documents:    {total_documents:,}")
print(f"Total tokens: {total_tokens:,}")
print(f"EOS tokens:   {total_documents:,}")
print(f"Output:       {OUTPUT_PATH}")
print(f"Size:         {OUTPUT_PATH.stat().st_size / 1024**3:.2f} GiB")