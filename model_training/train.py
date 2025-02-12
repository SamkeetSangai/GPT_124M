# Importing necessary libraries
import os
import math
import time
import torch
import random
import inspect
import logging
import tiktoken
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from transformers import GPT2LMHeadModel
from hellaswag import render_example, iterate_examples
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Torch Config
torch._dynamo.config.suppress_errors = True

# -----------------------------------------------------------------------------
# Configure logging
logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# -----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """
    This module performs multi-head self-attention in a causal (autoregressive) manner.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    A feed-forward network block used in Transformer architectures.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    A single Transformer block: LayerNorm -> CausalSelfAttention -> Residual -> LayerNorm -> MLP -> Residual
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """
    GPT configuration dataclass storing model hyperparameters.
    """

    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    batch_size: int = 0.5e6
    learning_rate: float = 6.0e-4


def get_gptconfig(model_type: str, **overrides) -> GPTConfig:
    """
    Returns a GPTConfig object for the specified model_type.
    """
    logging.info(f"Generating GPTConfig for model type: {model_type}")

    config_map = {
        "gpt2": dict(
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=1024,
            batch_size=0.5e6,
            learning_rate=6.0e-4,
        ),
        "gpt2-medium": dict(
            n_layer=24,
            n_head=16,
            n_embd=1024,
            block_size=1024,
            batch_size=0.5e6,
            learning_rate=3.0e-4,
        ),
        "gpt2-large": dict(
            n_layer=36,
            n_head=20,
            n_embd=1280,
            block_size=1024,
            batch_size=0.5e6,
            learning_rate=2.5e-4,
        ),
        "gpt2-xl": dict(
            n_layer=48,
            n_head=25,
            n_embd=1600,
            block_size=1024,
            batch_size=1e6,
            learning_rate=2.0e-4,
        ),
        "gpt3-small": dict(
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=2048,
            batch_size=0.5e6,
            learning_rate=6.0e-4,
        ),
        "gpt3-medium": dict(
            n_layer=24,
            n_head=16,
            n_embd=1024,
            block_size=2048,
            batch_size=0.5e6,
            learning_rate=3.0e-4,
        ),
        "gpt3-large": dict(
            n_layer=24,
            n_head=16,
            n_embd=1536,
            block_size=2048,
            batch_size=0.5e6,
            learning_rate=2.5e-4,
        ),
        "gpt3-xl": dict(
            n_layer=24,
            n_head=32,
            n_embd=2048,
            block_size=2048,
            batch_size=1e6,
            learning_rate=2.0e-4,
        ),  # Changing n_head from 24 → 34 to ensure that n_embd // n_head results in a whole number
        "gpt3-2.7B": dict(
            n_layer=32,
            n_head=32,
            n_embd=2560,
            block_size=2048,
            batch_size=1e6,
            learning_rate=1.6e-4,
        ),
        "gpt3-6.7B": dict(
            n_layer=32,
            n_head=32,
            n_embd=4096,
            block_size=2048,
            batch_size=2e6,
            learning_rate=1.2e-4,
        ),
        "gpt3-13B": dict(
            n_layer=40,
            n_head=40,
            n_embd=5140,
            block_size=2048,
            batch_size=2e6,
            learning_rate=1.0e-4,
        ),
        "gpt3-175B": dict(
            n_layer=96,
            n_head=96,
            n_embd=12288,
            block_size=2048,
            batch_size=3.2e6,
            learning_rate=0.6e-4,
        ),
    }

    if model_type not in config_map:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Choose from {list(config_map.keys())}."
        )

    cfg_dict = config_map[model_type]
    cfg_dict["vocab_size"] = 50257  # Default

    # Apply user overrides if any
    cfg_dict.update(overrides)
    return GPTConfig(**cfg_dict)


class GPT(nn.Module):
    """
    The GPT language model class:
      - Embeddings (token + positional)
      - A stack of Transformer blocks
      - A final LayerNorm + linear head (lm_head)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        gpt2_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        if model_type not in gpt2_models:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Only GPT-2 versions are allowed: {gpt2_models}."
            )

        logging.info(f"Loading weights from pretrained GPT: {model_type}")
        config = get_gptconfig(model_type)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        logging.info("Successfully loaded pretrained weights.")
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        if master_process:
            logging.info(f"Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


# -----------------------------------------------------------------------------


def load_tokens(filename):
    """
    Utility to load tokens from a .npy file into a torch tensor (long dtype).
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    """
    A simple data loader that yields contiguous sequences of tokens
    for training or validation, given a set of shard files.
    """

    def __init__(self, B, T, process_rank, num_processes, split, data_root):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"

        self.reset()

    def reset(self):
        """Reset to the first shard and set the current position based on process rank."""
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """
        Return the next batch of token indices (x, y) where x is input and y is target.
        """
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------


def get_most_likely_row(tokens, mask, logits):
    """
    Given tokens, mask, and logits, returns the index of the completion
    with the lowest average loss over the masked positions.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)

    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# -----------------------------------------------------------------------------
# DDP detection
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp and torch.cuda.is_available():
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

device_type = "cuda" if device.startswith("cuda") else "cpu"

# Performance Settings
if device_type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# Set Seeds
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# GPT Tiktoken Encoder
enc = tiktoken.get_encoding("gpt2")


def nearest_divisible_value(desired_value, multiple):
    """
    Finds the smallest number >= desired_value that is divisible by both `multiple` and `2`.
    """
    multiple = multiple if multiple % 2 == 0 else multiple * 2
    next_value = math.ceil(desired_value / multiple) * multiple
    return next_value


# -----------------------------------------------------------------------------
# GPT Config
model_type = "gpt2"
vocab_size = 50304
gptconfig = get_gptconfig(model_type=model_type, vocab_size=vocab_size)

# Desired training configuration
desired_batch_size = gptconfig.batch_size
B = 12
T = gptconfig.block_size
total_batch_size = nearest_divisible_value(
    desired_batch_size, multiple=B * T * ddp_world_size
)
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

# Data Loader
data_root = "edu_fineweb10B"
train_loader = DataLoaderLite(
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="train",
    data_root=data_root,
)
val_loader = DataLoaderLite(
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="val",
    data_root=data_root,
)

torch.set_float32_matmul_precision("high")

# -----------------------------------------------------------------------------
# Model Setup
model = GPT(gptconfig)
model.to(device)
# model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# -----------------------------------------------------------------------------
# Optimizer and LR Schedule
max_lr = gptconfig.learning_rate * 3
min_lr = max_lr * 0.1
token_size = int(data_root[len("edu_fineweb") : -1]) * 1e9
num_epochs = 2
max_steps = int((token_size / total_batch_size) * num_epochs)
warmup_steps = 100  # Warm-up Steps


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=max_lr, device_type=device_type
)

# -----------------------------------------------------------------------------
# Checkpoint directory and resume logic
checkpoint_dir = "checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)
persistent_checkpoint_path = os.path.join(checkpoint_dir, f"model_{model_type}.pt")

start_step = 0

# Resume training from a checkpoint if it exists and resumption is enabled
resume = True
if os.path.isfile(persistent_checkpoint_path) and resume:
    if master_process:
        logging.info(f"Resuming from checkpoint: {persistent_checkpoint_path}")
    ckpt_data = torch.load(persistent_checkpoint_path, map_location=device)

    # Load model/optimizer
    raw_model.load_state_dict(ckpt_data["model_state_dict"])
    optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
    # start_step = ckpt_data["step"]

    # Restore dataloader state
    train_loader.current_shard = ckpt_data["train_loader_state"]["current_shard"]
    train_loader.current_position = ckpt_data["train_loader_state"]["current_position"]

    if master_process:
        logging.info(f"Resumed training from step {start_step}")

# -----------------------------------------------------------------------------

# Interval Parameters
val_steps = 250  # Perform validation every 'val_steps' training iterations
checkpoint_multiplier = 4  # Checkpoint interval as a multiple of val_steps
hellaswag_steps = 500  # Perform HellaSwag evaluation every 'hellaswag_steps' iterations
sample_steps = 1000  # Generate text samples every 'sample_steps' iterations


# -----------------------------------------------------------------------------
# Printing All Variables


def print_var_values():
    vars = {
        # DDP Configuration
        "DDP Enabled": ddp,
        "DDP Rank": ddp_rank,
        "DDP Local Rank": ddp_local_rank,
        "DDP World Size": ddp_world_size,
        "Master Process": master_process,
        "Device Type": device_type,
        # Model & Training Config
        "Max Learning Rate": max_lr,
        "Min Learning Rate": min_lr,
        "Data Token Size (B)": token_size / 1e9,
        "Num Epochs": num_epochs,
        "Max Steps": max_steps,
        "Warm-up Steps": warmup_steps,
        # Training Configuration
        "Vocab Size": vocab_size,
        "Batch Size (from GPT Config)": gptconfig.batch_size,
        "Block Size (T)": gptconfig.block_size,
        "Total Batch Size": total_batch_size,
        "Gradient Accumulation Steps": grad_accum_steps,
        # DataLoader Config
        "Batch Size (B)": B,
        "Data Root": data_root,
        # Training Resumption
        "Start Step": start_step,
        # Interval Settings
        "Validation Interval": val_steps,
        "Checkpoint Multiplier": checkpoint_multiplier,
        "HellaSwag Evaluation Interval": hellaswag_steps,
        "Sample Generation Interval": sample_steps,
        # GPT Configuration
        "GPT Model Type": model_type,
        "Number of Layers (n_layer)": gptconfig.n_layer,
        "Number of Heads (n_head)": gptconfig.n_head,
        "Embedding Size (n_embd)": gptconfig.n_embd,
        "Block Size (context length)": gptconfig.block_size,
        "Batch Size": gptconfig.batch_size,
        "Learning Rate": gptconfig.learning_rate,
    }

    # Print General Configuration
    print("\n" + "=" * 60)
    print(" Configuration Summary ".center(60, "="))
    print("=" * 60 + "\n")

    for key, value in vars.items():
        print(f"{key.ljust(40)} : {value}")

    # Print GPT Configuration
    print("\n" + "=" * 60)


# DEBUG
print_var_values()

# Training Loop
for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    # Evaluate validation loss periodically
    if step % val_steps == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x_cpu, y_cpu = val_loader.next_batch()
                if device_type == "cuda":
                    x_cpu, y_cpu = x_cpu.pin_memory(), y_cpu.pin_memory()
                x, y = x_cpu.to(device, non_blocking=True), y_cpu.to(
                    device, non_blocking=True
                )
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            val_loss_value = val_loss_accum.item()
            logging.info(f"Step {step}: validation loss = {val_loss_value:.4f}")

            # Save checkpoint periodically or at the last step
            if step > 0 and (
                step % (val_steps * checkpoint_multiplier) == 0 or last_step
            ):
                # Prepare checkpoint data
                persistent_checkpoint = {
                    "model_state_dict": raw_model.state_dict(),
                    "model_config": raw_model.config,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "train_loader_state": {
                        "current_shard": train_loader.current_shard,
                        "current_position": train_loader.current_position,
                    },
                }
                torch.save(persistent_checkpoint, persistent_checkpoint_path)

                # A smaller rotating checkpoint for additional tracking
                rotating_checkpoint_path = os.path.join(
                    checkpoint_dir, f"step_{step:05d}.pt"
                )
                rotating_checkpoint = {
                    "step": step,
                    "val_loss": val_loss_value,
                    "device": device,
                    "world_size": ddp_world_size,
                }
                torch.save(rotating_checkpoint, rotating_checkpoint_path)

                logging.info(f"Checkpoint saved at step {step}")

    # Evaluate HellaSwag periodically
    if step % hellaswag_steps == 0 or last_step:
        model.eval()
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(
                num_correct_norm, dtype=torch.long, device=device
            )
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()

        acc_norm = num_correct_norm / num_total
        if master_process:
            logging.info(f"Step {step}: HellaSwag Accuracy = {acc_norm:.4f}")

    # Generate text samples periodically
    if (step > 0 and step % sample_steps == 0) or last_step:
        model.eval()
        num_return_sequences = 2
        max_length = 32
        prompt_text = "Hello, I'm a language model,"
        tokens = enc.encode(prompt_text)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        tokens = tokens.repeat(num_return_sequences, 1).to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)

        while tokens.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = model(tokens)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                tokens = torch.cat((tokens, xcol), dim=1)

        for i in range(num_return_sequences):
            out_tokens = tokens[i, :max_length].tolist()
            decoded = enc.decode(out_tokens)
            logging.info(f"Step {step}, Rank {ddp_rank}, Sample {i}: {decoded}")

    # Training Step
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x_cpu, y_cpu = train_loader.next_batch()
        if device_type == "cuda":
            x_cpu, y_cpu = x_cpu.pin_memory(), y_cpu.pin_memory()
        x, y = x_cpu.to(device, non_blocking=True), y_cpu.to(device, non_blocking=True)

        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()

    # Compute time taken for the current step
    t1 = time.time()
    dt = t1 - t0  # Time taken for this step
    tokens_processed = (
        train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    )
    tokens_per_sec = tokens_processed / dt

    # Estimate total training time based on the last step duration
    remaining_steps = max_steps - step
    estimated_total_time_hours = (
        dt * remaining_steps
    ) / 3600  # Convert seconds to hours

    if master_process:
        logging.info(
            f"Step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | "
            f"grad_norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | "
            f"ETA (All Epochs): {estimated_total_time_hours:.2f} Hours"
        )

if ddp:
    destroy_process_group()
