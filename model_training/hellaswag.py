"""
Downloads and evaluates HellaSwag in Python.
Repository: https://github.com/rowanz/hellaswag

Example HellaSwag JSON item:

{
    "ind": 24,
    "activity_label": "Roof shingle removal",
    "ctx_a": "A man is sitting on a roof.",
    "ctx_b": "he",
    "ctx": "A man is sitting on a roof. he",
    "split": "val",
    "split_type": "indomain",
    "label": 3,
    "endings": [
        "is using wrap to wrap a pair of skis.",
        "is ripping level tiles off.",
        "is holding a rubik's cube.",
        "starts pulling up roofing on a roof."
    ],
    "source_id": "activitynet~v_-JhWjGDPHMY"
}

Field descriptions:
- **ind**: Dataset ID.
- **activity_label**: The ActivityNet or WikiHow label for this example.
- **context**: The full context is in `ctx`. If `ctx_b` is nonempty, then `ctx` consists of `ctx_a` followed by a space and `ctx_b`.
- **endings**: A list of 4 possible endings. The correct answer is given by `label` (0,1,2, or 3).
- **split**: Indicates whether the example is from the train, val, or test set.
- **split_type**: "indomain" if the activity label was seen during training, otherwise "zeroshot".
- **source_id**: The originating video or WikiHow article.

GPT-2 (124M) Accuracy:
- Eleuther Harness: Accuracy = 28.92%, Normalized Accuracy = 31.14% (multiple-choice style).
- This script: Accuracy = 28.59%, Normalized Accuracy = 29.55% (completion style).

GPT-2 XL (1558M) Accuracy:
- Eleuther Harness: Accuracy = 40.04%, Normalized Accuracy = 50.89% (multiple-choice style).
- This script: Accuracy = 38.42%, Normalized Accuracy = 48.93% (completion style).

The validation set of HellaSwag contains a total of 10,042 examples.
"""

# Importing necessary libraries
import os
import json
import torch
import logging
import requests
import tiktoken
from tqdm import tqdm
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# -----------------------------------------------------------------------------
# Configure logging
logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# -----------------------------------------------------------------------------
# Define data cache directory
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")


# Function to download a file from a given URL
def download_file(url: str, fname: str, chunk_size=1024):
    """Downloads a file in chunks and displays a progress bar."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


# URLs for the HellaSwag dataset
hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# Load GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")


# Function to download dataset if not already present
def download(split):
    """Downloads the HellaSwag dataset and saves it locally."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)


# Function to process an example into tokenized format
def render_example(example):
    """Converts an example into tokenized tensors for model evaluation."""
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # Data for reproducibility
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # Encode the context
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []

    # Process each ending option
    for end in endings:
        end_tokens = enc.encode(" " + end)  # Prepend space for GPT-2 tokenization
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # Normalize tensor shapes
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


# Function to iterate over dataset examples
def iterate_examples(split):
    """Reads dataset line by line and yields each example."""
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


@torch.no_grad()
def evaluate(model_type, device):
    """Evaluates the model on HellaSwag validation set."""
    torch.set_float32_matmul_precision("high")  # Use tf32 for performance

    # Load pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Compute model logits
        logits = model(tokens).logits

        # Shift logits and tokens for loss computation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()

        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(
            flat_shift_logits, flat_shift_tokens, reduction="none"
        )
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # Compute average loss for completion region
        shift_mask = mask[..., 1:].contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # Determine predicted index
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # Update evaluation statistics
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        # Log evaluation results
        logging.info(
            f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}"
        )

        # # Debug: Display first few examples
        # if num_total < 10:
        #     print("---")
        #     print(f"Context:\n {example['ctx']}")
        #     print("Endings:")
        #     for i, end in enumerate(example["endings"]):
        #         print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
        #     print(f"predicted: {pred_norm}, actual: {label}")


if __name__ == "__main__":
    import argparse

    # Argument parser for CLI execution
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_type", type=str, default="gpt2-xl", help="The model type to use"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="The device to use"
    )
    args = parser.parse_args()

    # Run Evaluation
    evaluate(args.model_type, args.device)
