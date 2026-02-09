---
library_name: transformers
license: mit
datasets:
- HuggingFaceFW/fineweb-edu
language:
- en
base_model:
- openai-community/gpt2
pipeline_tag: text-generation
tags:
- GPT
- GPT-3 Small
- GPT-3 Medium
- GPT-3 Large
- GPT-3 XL
- GPT-3 2.7B
- GPT-3 6.7B
- GPT-3 13B
- GPT-3 175B
- GPT-3
- GPT-2
- GPT-2 124M
- transformers
- mit
- HuggingFace
- fineweb-edu
- Decoder-Only
---
# Model Card for GPT-124M

## Overview

GPT-124M is a decoder-only transformer model based on OpenAI’s GPT-2 architecture. It is trained for text generation and other natural language processing (NLP) tasks. The model is designed for general-purpose language modeling, making it useful for applications such as text completion.

- **Library:** 🤗 `transformers`
- **License:** MIT
- **Datasets:** `HuggingFaceFW/fineweb-edu`
- **Language:** English
- **Base Model:** `openai-community/gpt2`
- **Pipeline Tag:** `text-generation`
- **Developer:** Samkeet Sangai
- **Funded By:** Samkeet Sangai
- **Shared By:** Samkeet Sangai
- **Model Type:** GPT Decoder-Only

## Model Sources

- **Paper:** [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **Paper:** [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)
- **Paper:** [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)
- **Video:** [Andrej Karpathy-Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU?si=KAo1y9dHYQAGJmj5)
- **Demo:** [GPT 124M Demo](https://huggingface.co/spaces/samkeet/GPT_124M)

## Model Details

### Model Description
GPT-124M is a lightweight generative language model fine-tuned on the `fineweb-edu` dataset. It can generate coherent and contextually relevant text but is not fine-tuned for instruction-following, safety, or factual accuracy.

### Training Configuration
- **Block Size:** `1024`
- **Vocabulary Size:** `50304`
- **Number of Layers:** `12`
- **Number of Attention Heads:** `12`
- **Embedding Size:** `768`
- **Hardware:** `8x NVIDIA RTX 4090 GPUs`
- **Training Duration:** `13 hours`
- **Dataset:** `fineweb-edu` (10 billion tokens)
- **Training Date:** `January 2025`
- **Validation Dataset:** 100 million tokens of HuggingFaceFW/fineweb-edu
  
## Usage
You can use this model for text generation using the `transformers` library.

### Method 1: Using Pipeline
```python
# Import necessary modules from transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load tokenizer and model
model_name = "samkeet/GPT_124M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Create text generation pipeline
pipe = pipeline("text-generation", model=model_name, tokenizer=tokenizer, trust_remote_code=True, device="cpu")

# Generate text
result = pipe("Earth revolves around the", do_sample=True, max_length=40, temperature=0.9, top_p=0.5, top_k=50)
print("Pipeline Output:", result)
```

### Method 1: Direct Generation
```python
# Import necessary libraries
import torch

# Function for direct tokenization and text generation
def generate_text(input_text, device='cpu'):
    tokens = tokenizer.encode(input_text, return_tensors='pt').to(device)
    model.to(device)
    
    # Generate output
    output = model.generate(
        tokens, 
        do_sample=True, 
        max_length=40, 
        temperature=0.9,
        top_p=0.5,
        top_k=50,
    )
    
    # Decode generated text
    generated_sentence = tokenizer.decode(output)
    return generated_sentence

# Generate text
input_text = "Earth revolves around the"
print("Direct Output:", generate_text(input_text))
```

### Fine-tuning & Downstream Use
This model can be fine-tuned for specific NLP applications like:
- Dialogue generation
- Text summarization
- Creative writing
- Code generation

## Limitations & Risks

### Out-of-Scope Use
- The model is **not instruction-tuned** for safety, ethics, or factual accuracy.
- It may produce **biased, misleading, or unsafe outputs**.
- It should **not** be used for tasks requiring high reliability, such as medical, legal, or financial applications.

### Bias, Risks, and Limitations
- The dataset may contain biases present in public web data.
- The model does not filter or detect offensive content.
- The model may **hallucinate** incorrect facts.

### Recommendations
- Always **verify** generated content before use.
- Implement **content filtering mechanisms** for deployment.
- Use in supervised environments only.

## Evaluation

### Training & Validation Loss
Validation was conducted using `100 million tokens` from the `HuggingFaceFW/fineweb-edu` dataset. The training and validation loss graph indicates a stable convergence with minimal overfitting. The training loss achieved a minimum value of 2.88, while the validation loss stabilized at 2.97.
![image/png](https://cdn-uploads.huggingface.co/production/uploads/670142e648894dfbedacacaf/fAwiSHr4f9SmO9PYiCntY.png)

### Results
The model was benchmarked against OpenAI’s GPT-2 Small and GPT-3 Small (both ~124M parameters). Remarkably, despite being trained on only `10 billion tokens`, compared to GPT-3 Small's `300 billion tokens`, GPT-124M was able to outperform both models in `HellaSwag` evaluation. This performance advantage is attributed to the specialized training data (educational content), which contrasts with GPT-3 Small’s broader multilingual and multi-domain training data.

According to Chinchilla’s scaling laws, an optimal token-to-parameter ratio suggests that a 124M-parameter model ideally requires `2.48 billion tokens` for training. The excess training tokens used in GPT-3 Small might have led to diminishing returns in performance.
![image/png](https://cdn-uploads.huggingface.co/production/uploads/670142e648894dfbedacacaf/Ne2MYAB2C0yHWFJLjCww3.png)

### Key Insights from Evaluation
- **Efficient Training:** The model demonstrates impressive performance relative to its training token count, suggesting an efficient use of resources due to training using the Distributed Data Parallel (DDP) technique.
- **Data-Specific Advantage:** Training exclusively on educational data may have given GPT-124M an edge in evaluation metrics like `HellaSwag`.
- **Scaling Considerations:** GPT-3 Small, despite being trained on 300B tokens, does not exhibit proportionally better performance due to scaling limitations.

## Environmental Impact

- **Hardware Used:** `8x NVIDIA RTX 4090 GPUs`
- **Training Time:** `13 hours -> 104 GPU hours`
- **Estimated Carbon Emissions:** `13.48 kg CO2 eq.`
- **Equivalent to:**
  - `54.5 km` driven by an average ICE car
  - `6.75 kg` of coal burned
  - `0.22` tree seedlings sequestering carbon for 10 years

## Technical Specifications

### Model Architecture
GPT-124M follows the architecture of OpenAI's GPT-2, which consists of:
- **Transformer-based decoder model**
- **Self-attention mechanism**
- **Layer normalization & feed-forward networks**

### Compute Infrastructure
- **Hardware:** 8x NVIDIA RTX 4090 GPUs
- **Software:** PyTorch, Hugging Face Transformers
- **Precision:** FP32

Gotcha — here’s a **tight, concise section** you can drop in **as-is**.
It keeps only the essentials: **data, setup, choices, Kaggle**, no fluff.

---

## Instruction-Tuned Model

### Training Data

The instruction-tuned GPT-124M is fine-tuned on the **`tatsu-lab/alpaca`** dataset, containing high-quality instruction–response pairs across reasoning, explanation, summarization, and creative tasks. Samples are **length-filtered** to fit the 1024-token context window, counting instruction, input, response, and EOS tokens.

### Prompt and Objective

Training follows an Alpaca-style format:

```
### Instruction:
<instruction and optional input>

### Response:
<target output>
```

Causal language modeling is used, with **loss applied only to response tokens** (prompt tokens masked with `-100`) and an explicit EOS token appended.

### Training Setup

* **Platform:** Kaggle (GPU-backed notebooks)
* **Framework:** PyTorch
* **Precision:** FP32
* **Optimizer:** AdamW with warmup + cosine decay
* **Stability:** Gradient clipping and fixed-length batching

### Fine-Tuning Choices

* Supports **full fine-tuning** and **LoRA-based parameter-efficient tuning**
* LoRA can be merged into base weights for a standalone instruct model
* Supervised fine-tuning (SFT) chosen for simplicity and reproducibility
* No RLHF or safety-specific tuning applied

### Outcome

Instruction tuning improves instruction following, output structure, and task performance while preserving the base model’s generative capabilities. The model remains non-aligned and may hallucinate.

## Citation

If you use this model, please cite:

```bibtex
@article{gpt124m,
  title={GPT-124M: A Compact Transformer Model for NLP},
  author={Samkeet Sangai},
  year={2024},
  url={https://huggingface.co/samkeet/GPT_124M}
}
```

## Contact
For inquiries, contact [Samkeet Sangai](https://www.linkedin.com/in/samkeet-sangai/).
