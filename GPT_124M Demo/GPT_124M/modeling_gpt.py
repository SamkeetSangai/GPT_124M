# Importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from .configuration_gpt import GPTConfig


class GPT(nn.Module):
    """
    The GPT language model:
    - Embeddings (token + positional)
    - Stack of Transformer blocks
    - Final LayerNorm + Linear head for output logits
    """

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
    ):
        super().__init__()

        # Store model hyperparameters
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        # Transformer components stored in a module dictionary
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.vocab_size, self.n_embd),  # Token embedding
                wpe=nn.Embedding(self.block_size, self.n_embd),  # Positional embedding
                h=nn.ModuleList(
                    [self.Block(self.n_embd, self.n_head) for _ in range(self.n_layer)]
                ),  # Transformer blocks
                ln_f=nn.LayerNorm(self.n_embd),  # Final layer normalization
            )
        )

        # Linear head for output logits
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        # Tie weights between token embedding and output projection
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, x):
        B, T = x.shape  # Batch size and sequence length
        assert T <= self.block_size, "Cannot forward sequence longer than block size"

        # Token and positional embeddings
        tok_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb.unsqueeze(0)

        # Forward pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)  # Final layer norm
        logits = self.lm_head(x)  # Compute logits
        return logits

    class CausalSelfAttention(nn.Module):
        """
        Multi-head self-attention with causal masking.
        """

        def __init__(self, n_embd, n_head):
            super().__init__()
            assert (
                n_embd % n_head == 0
            ), "Embedding dimension must be divisible by number of heads"
            self.n_head = n_head
            self.n_embd = n_embd

            # Linear layers for query, key, and value
            self.c_attn = nn.Linear(n_embd, 3 * n_embd)
            self.c_proj = nn.Linear(n_embd, n_embd)

        def forward(self, x):
            B, T, C = x.size()
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)

            # Reshape and transpose for multi-head attention
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            # Apply scaled dot-product attention with causal masking
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            # Reshape and apply output projection
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.c_proj(y)
            return y

    class MLP(nn.Module):
        """
        Feed-forward network block used in Transformer architectures.
        """

        def __init__(self, n_embd):
            super().__init__()
            self.c_fc = nn.Linear(n_embd, 4 * n_embd)
            self.gelu = nn.GELU(approximate="tanh")
            self.c_proj = nn.Linear(4 * n_embd, n_embd)

        def forward(self, x):
            return self.c_proj(self.gelu(self.c_fc(x)))

    class Block(nn.Module):
        """
        A single Transformer block.
        """

        def __init__(self, n_embd, n_head):
            super().__init__()
            self.ln_1 = nn.LayerNorm(n_embd)
            self.attn = GPT.CausalSelfAttention(n_embd, n_head)
            self.ln_2 = nn.LayerNorm(n_embd)
            self.mlp = GPT.MLP(n_embd)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


class GPTModelForTextGeneration(PreTrainedModel):
    """
    A wrapper class for GPT-based text generation.
    This integrates a Transformer model within the Hugging Face `PreTrainedModel` framework.
    """

    config_class = GPTConfig

    def __init__(self, config):
        super().__init__(config)

        # Instantiate the GPT model with the provided configuration
        self.model = GPT(
            block_size=config.block_size,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
        )

    def forward(self, input_ids: torch.Tensor):
        # Check input_ids type and shape
        assert isinstance(input_ids, torch.Tensor), "input_ids must be a PyTorch tensor"

        tokens = input_ids.clone()  # Avoid modifying input_ids directly
        tokens = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens

        assert (
            tokens.ndim == 2 and tokens.shape[0] == 1
        ), "input_ids must have 2 dimensions: (1, sequence_length)"

        # Check token values
        assert torch.all(
            (tokens >= 0) & (tokens <= self.model.vocab_size)
        ), "input_ids contain invalid token values"

        # Forward pass through the model
        logits = self.model.forward(tokens)

        return {"logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 0.9,
        device: str = "cpu",
    ):
        """
        Generates text using autoregressive sampling with top-k, top-p, and temperature.
        """

        # Validate device type
        if device.startswith("cuda"):
            assert torch.cuda.is_available(), "CUDA is not available, please use 'cpu'"
            if device != "cuda":  # Check for specific CUDA device (cuda:n)
                try:
                    device_index = int(device.split(":")[1])  # Extract device number
                    assert (
                        0 <= device_index < torch.cuda.device_count()
                    ), f"Invalid CUDA device index: {device_index}"
                except (IndexError, ValueError):
                    raise ValueError(
                        "Invalid device format. Use 'cpu', 'cuda', or 'cuda:N' where N is an integer."
                    )
        elif device != "cpu":
            raise ValueError("Invalid device. Use 'cpu', 'cuda', or 'cuda:N'.")

        # Move input tensor and model to the specified device
        input_ids = input_ids.to(device)
        self.model.to(device)

        # Check input_ids type and shape
        assert isinstance(input_ids, torch.Tensor), "input_ids must be a PyTorch tensor"
        tokens = input_ids.clone()  # Avoid modifying input_ids directly
        tokens = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens

        assert (
            tokens.ndim == 2 and tokens.shape[0] == 1
        ), "input_ids must have 2 dimensions: (1, sequence_length)"

        # Check token values
        assert torch.all(
            (tokens >= 0) & (tokens < self.model.vocab_size)
        ), "input_ids contain invalid token values"

        # Check max_length
        assert (
            isinstance(max_length, int) and max_length >= 1
        ), "max_length must be a positive integer"
        assert (
            max_length <= self.model.block_size
        ), f"max_length must be in range [1, {self.model.block_size}]"

        # Check top_k
        assert isinstance(top_k, int) and top_k >= 1, "top_k must be a positive integer"

        # Check top_p
        assert (
            isinstance(top_p, (int, float)) and 0.0 <= top_p <= 1.0
        ), "top_p must be in range [0, 1]"

        # Check temperature
        assert (
            isinstance(temperature, (int, float)) and 0.0 <= temperature <= 1.0
        ), "temperature must be in range [0, 1]"

        # Move tokens to the correct device
        tokens = tokens.to(device)

        # Autoregressive token generation loop
        while tokens.size(1) < max_length:
            logits = self.forward(tokens)["logits"][:, -1, :]
            logits = logits / max(0.01, temperature)

            if do_sample:
                top_k = min(top_k, logits.size(-1))  # Safety check

                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = (
                    logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
                )
                logits[indices_to_remove] = float("-inf")

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)

                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Replace logits to be removed with -inf in the sorted_logits
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                # Then reverse the sorting process by mapping back sorted_logits to their original position
                logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

                # Convert sorted indices back to original vocab indices
                next_tokens = torch.multinomial(F.softmax(logits, -1), 1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            tokens = torch.cat((tokens, next_tokens), dim=1)

        return tokens.flatten()
