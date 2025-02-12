# Importing necessary libraries
import torch
from huggingface_hub import login
from gpt_model.configuration_gpt import GPTConfig
from transformers.pipelines import PIPELINE_REGISTRY
from gpt_model.pipeline_gpt import GPT124MTextGenerationPipeline
from gpt_model.modeling_gpt import GPTModelForTextGeneration, GPT
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline

# Hugging Face Hub Login
huggingface_hub_access_token = "hf_***********************************"
login(huggingface_hub_access_token)

# Define Model Configuration
model_config = {
    "block_size": 1024,
    "vocab_size": 50304,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
}

# Save and Load GPT Configuration
gpt_config = GPTConfig(**model_config)
gpt_config.save_pretrained("GPT_124M")
gpt_config = GPTConfig.from_pretrained("GPT_124M")

# Initialize GPT Model
gpt_model = GPTModelForTextGeneration(gpt_config)

# Load Pretrained Model Checkpoint
checkpoint = torch.load(
    "./GPT_pretrained_model/model_gpt2.pt", map_location=torch.device("cpu")
)

# Initialize Pretrained Model
pretrained_model = GPT(**model_config)
pretrained_model.load_state_dict(checkpoint["model_state_dict"])
pretrained_model.to("cpu")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("GPT_124M")
tokenizer = AutoTokenizer.from_pretrained("GPT_124M")

# Transfer Weights from Pretrained Model to GPT Model
gpt_model.model.load_state_dict(pretrained_model.state_dict())

# Register Custom Model and Configuration with Transformers
AutoConfig.register("custom_gpt", GPTConfig)
AutoModelForCausalLM.register(GPTConfig, GPTModelForTextGeneration)
GPTConfig.register_for_auto_class("AutoConfig")
GPTModelForTextGeneration.register_for_auto_class("AutoModelForCausalLM")

# Push Model and Tokenizer to Hugging Face Hub
gpt_model.push_to_hub("samkeet/GPT_124M", safe_serialization=False)
tokenizer.push_to_hub("samkeet/GPT_124M", safe_serialization=False)

# Register Custom Text Generation Pipeline
PIPELINE_REGISTRY.register_pipeline(
    "text-generation",
    pipeline_class=GPT124MTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
    default={"pt": ("samkeet/GPT_124M")},
    type="text",
)

# Initialize Custom Text Generation Pipeline
gpt124m_pipline = pipeline(
    task="text-generation",
    model=gpt_model,
    tokenizer=tokenizer,
    framework="pt",
    device="cpu",
    trust_remote_code=True,
    pipeline_class=GPT124MTextGenerationPipeline,
)

# Push Pipeline to Hugging Face Hub
gpt124m_pipline.push_to_hub("samkeet/GPT_124M", safe_serialization=False)
