# Importing necessary libraries
import os
import logging
import tiktoken
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from datasets import load_dataset

# -----------------------------------------------------------------------------
# Configure logging
logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# -----------------------------------------------------------------------------
# Configuration settings
REMOTE_NAME = "sample-10BT"
SHARD_SIZE = int(1e8)  # 100M tokens per shard
TOTAL_TOKENS_TARGET = int(10 * 1e9)  # Tokens target
LOCAL_DIR = f"edu_fineweb{int(TOTAL_TOKENS_TARGET//1e9)}B"

# Create the local directory if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), LOCAL_DIR)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
logging.info(f"Data cache directory set to: {DATA_CACHE_DIR}")

# Load the dataset
fw = load_dataset(
    "HuggingFaceFW/fineweb-edu", name=REMOTE_NAME, split="train", streaming=True
)
logging.info("Dataset loaded successfully.")

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
EOT_TOKEN = enc._special_tokens["<|endoftext|>"]  # End of text token
logging.info("Tokenizer initialized.")


def tokenize(doc):
    """Tokenizes a single document and returns a numpy array of uint16 tokens."""
    tokens = [EOT_TOKEN]  # Special token to delimit documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "Token dictionary too large for uint16"
    return tokens_np


def write_datafile(filename, tokens_np):
    """Saves tokenized data to a .npy file."""
    np.save(filename, tokens_np)
    logging.info(f"Shard saved: {filename} ({tokens_np.size} tokens)")


if __name__ == "__main__":
    num_processes = max(1, os.cpu_count() // 2)
    total_tokens_written = 0  # Track total tokens processed
    shard_index = 0  # Track the number of shards created

    logging.info(f"Starting tokenization using {num_processes} processes.")

    # Initialize multiprocessing pool
    with mp.Pool(num_processes) as pool:
        token_buffer = np.empty((SHARD_SIZE,), dtype=np.uint16)
        token_count = 0
        progress_bar = tqdm(
            total=TOTAL_TOKENS_TARGET, unit="tokens", desc="Processing Tokens"
        )

        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if total_tokens_written >= TOTAL_TOKENS_TARGET:
                logging.info("Tokenization target reached. Stopping process.")
                break

            if token_count + len(tokens) < SHARD_SIZE:
                token_buffer[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
                )
                remainder = SHARD_SIZE - token_count
                token_buffer[token_count : token_count + remainder] = tokens[:remainder]
                write_datafile(filename, token_buffer)
                total_tokens_written += SHARD_SIZE
                shard_index += 1

                # Reset buffer and store remaining tokens in new shard
                token_buffer[: len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

            progress_bar.update(len(tokens))
            logging.info(f"Processed {total_tokens_written} tokens so far.")

        if token_count > 0 and total_tokens_written < TOTAL_TOKENS_TARGET:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
            )
            write_datafile(filename, token_buffer[:token_count])
            total_tokens_written += token_count

        progress_bar.close()

    logging.info(f"Tokenization complete. Total tokens written: {total_tokens_written}")
    print(f"Tokenization complete. Total tokens written: {total_tokens_written}")
