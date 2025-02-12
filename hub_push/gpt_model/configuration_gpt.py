# Importing libraries
from transformers import PretrainedConfig


class GPTConfig(PretrainedConfig):
    model_type = "custom_gpt"

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        **kwargs,
    ):
        """
        GPT configuration dataclass storing model hyperparameters.
        """
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer: int = n_layer
        self.n_head: int = n_head
        self.n_embd: int = n_embd
