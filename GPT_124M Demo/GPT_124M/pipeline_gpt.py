# Importing necessary libraries
import torch
from transformers import Pipeline
from .modeling_gpt import GPTModelForTextGeneration


class GPT124MTextGenerationPipeline(Pipeline):

    def _sanitize_parameters(self, **kwargs):
        """
        Organizes and sanitizes input parameters into separate dictionaries for:
        - Preprocessing (encoding)
        - Model forward pass (generation settings)
        - Postprocessing (decoding)
        """

        preprocess_kwargs = {}
        forward_kwargs = {
            "max_length": kwargs.get("max_length", 50),
            "do_sample": kwargs.get("do_sample", True),
            "top_k": kwargs.get("top_k", 50),
            "top_p": kwargs.get("top_p", 0.95),
            "temperature": kwargs.get("temperature", 0.9),
            "device": kwargs.get("device", self.device.type),
        }
        postprocess_kwargs = {}

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, prompt_text: str, **preprocess_kwargs):
        """
        Encodes input text into token IDs using the tokenizer and converts it to a PyTorch tensor.
        """

        assert (
            isinstance(prompt_text, str) and len(prompt_text) > 0
        ), "prompt_text must be a non-empty string"

        # Encode the input text
        input_ids = self.tokenizer.encode(prompt_text)

        # Convert to a PyTorch tensor
        input_tensor = torch.tensor([input_ids])

        return {"input_ids": input_tensor}

    def _forward(self, model_inputs, **forward_kwargs):
        """
        Forwards the tokenized input to the model's generate method.
        """

        return self.model.generate(**model_inputs, **forward_kwargs)

    def postprocess(self, model_output, **postprocess_kwargs):
        """
        Decodes token ID into human-readable text using the tokenizer.
        """

        return self.tokenizer.decode(model_output)
