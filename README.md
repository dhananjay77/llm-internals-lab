# LLM Internals Lab

A hands-on learning repository for understanding how Large Language Models work — from tokenization and embeddings to Transformer internals, training, LoRA, quantization, and eventually building a Transformer from scratch.

The goal is not just to use an LLM, but to understand **what happens inside the model and why**.

---

## Learning Journey

```text
Text
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Embeddings
  ↓
Transformer
  ├── RMSNorm
  ├── Q / K / V
  ├── RoPE
  ├── Grouped Query Attention
  ├── Output Projection
  ├── Residual Connection
  ├── RMSNorm
  └── SwiGLU MLP
  ↓
LM Head
  ↓
Logits
  ↓
Probabilities
  ↓
Next Token
```

During training:

```text
Text
  ↓
Tokens
  ↓
Transformer
  ↓
Logits
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Optimizer
  ↓
Updated Weights
  ↓
Repeat
```

---

# Part 1 — Understand an Existing LLM

The first stage uses **SmolLM2-135M** to inspect the internals of a real causal language model.

## Model

```text
HuggingFaceTB/SmolLM2-135M
```

Approximate architecture:

```text
Vocabulary        : 49,152
Hidden size       : 576
Transformer layers: 30
Attention heads   : 9
KV heads          : 3
Head dimension    : 64
MLP intermediate  : 1,536
Context length     : 8,192
```

The model uses:

* Causal self-attention
* Grouped Query Attention (GQA)
* RoPE positional encoding
* RMSNorm
* SwiGLU MLP
* Weight-tied embeddings / LM head

---

# Part 2 — Transformer Internals

The experiments progressively inspect the Transformer computation.

## Attention

```text
Q = XWq
K = XWk
V = XWv

Attention(Q,K,V)
    = softmax(QKᵀ / √d)V
```

The experiments inspect:

* Q/K/V projections
* Attention scores
* Causal masking
* Softmax attention weights
* Multiple attention heads
* GQA
* Value-weighted attention output
* Output projection
* Residual connections

## RoPE

Rotary positional embeddings are used to inject positional information into Q and K.

The experiments demonstrate how rotating Q/K changes their attention compatibility while preserving dimensionality.

## Transformer Block

The model uses an architecture conceptually similar to:

```text
Input
  ↓
RMSNorm
  ↓
Self Attention
  ↓
Residual
  ↓
RMSNorm
  ↓
SwiGLU MLP
  ↓
Residual
  ↓
Output
```

---

# Part 3 — From Hidden States to Predictions

The final hidden representation is projected through the language-model head:

```text
Hidden state
576 dimensions
     ↓
LM Head
     ↓
49,152 logits
     ↓
Softmax
     ↓
Token probabilities
```

The inference experiments demonstrate:

* Greedy decoding
* Probability distributions
* Sampling
* Temperature
* Next-token generation

---

# Part 4 — Training Mechanics

Training experiments demonstrate the complete learning loop.

## Loss

For the correct next token:

```text
Loss = -log(P(correct token))
```

For causal language modeling:

```text
shift_logits = logits[:, :-1, :]
shift_labels = input_ids[:, 1:]
```

## Backpropagation

```text
Loss
  ↓
Backward
  ↓
Gradients
  ↓
Optimizer
  ↓
Weight update
```

The experiments inspect actual gradients and parameter changes.

## Training Loop

A small domain corpus is used to demonstrate:

```text
Forward
→ Loss
→ Zero gradients
→ Backward
→ Optimizer step
→ Repeat
```

---

# Part 5 — Domain Adaptation

A small software-engineering corpus is used for experimentation.

Topics include:

* Domain-Driven Design
* Entities
* Value Objects
* Aggregates
* Bounded Contexts
* Event-driven architecture
* Microservices
* Kubernetes
* Containers
* Observability
* Distributed tracing

This is intentionally a **toy dataset for learning**, not a production training corpus.

---

# Part 6 — Fine-Tuning and LoRA

The project demonstrates domain adaptation using instruction fine-tuning.

## Fine-Tuning

Instruction/response examples are used to teach the model a specific task or behavior.

## LoRA

Low-Rank Adaptation keeps the base model frozen and learns a small update:

```text
W' = W + ΔW

ΔW = B × A
```

For the experiment:

```text
Rank (r) = 8
Target modules = q_proj, v_proj
```

Approximately:

```text
Base model       ≈ 135M parameters
LoRA parameters  ≈ 461K
Trainable        ≈ 0.34%
```

The experiments inspect:

* LoRA parameter shapes
* Actual LoRA values
* Weight updates
* Effective weight changes
* Frozen base parameters
* LoRA training

---

# Part 7 — Quantization

Quantization reduces the number of bits used to represent model weights.

Conceptually:

```text
FP32 → BF16 → INT8 → INT4
```

Approximate weight memory for this model:

```text
FP32 : ~0.50 GB
BF16 : ~0.25 GB
INT8 : ~0.13 GB
INT4 : ~0.06 GB
```

The experiments demonstrate:

* INT8 quantization
* INT4 quantization
* Dequantization
* Quantization error
* Reduced numerical precision

These are educational simulations rather than optimized production quantization implementations.

---

# Part 8 — QLoRA

QLoRA combines:

```text
Quantized frozen base model
          +
Trainable LoRA adapter
```

Conceptually:

```text
          ┌──────────────────┐
          │  INT4 Base Model │
          │     Frozen       │
          └────────┬─────────┘
                   │
                   +
                   │
          ┌────────▼─────────┐
          │   LoRA Adapter   │
          │    Trainable     │
          └────────┬─────────┘
                   │
                   ▼
            Effective Model
```

The repository contains a conceptual QLoRA simulation to make this relationship explicit.

---

# Repository Structure

The experiments are organized by learning concept.

```text
llm-internals-lab/
│
├── README.md
├── .gitignore
│
├── data/
│   ├── pretrain.txt
│   └── finetune.jsonl
│
├── 01_tokenization/
├── 02_embeddings/
├── 03_attention/
│   ├── attention_demo.py
│   ├── attention_heads.py
│   ├── attention_output_demo.py
│   └── compare_q.py
│
├── 04_rope/
│   └── rope_demo.py
│
├── 05_transformer_block/
│   └── mlp_demo.py
│
├── 06_lm_head/
│   ├── lm_head_demo.py
│   ├── inference_demo.py
│   └── sampling_demo.py
│
├── 07_loss_backprop/
│   ├── loss_demo.py
│   ├── backprop_demo.py
│   └── optimizer_step_demo.py
│
├── 08_training/
│   ├── pretrain.py
│   └── training_loop_demo.py
│
├── 09_lora/
│   ├── finetune.py
│   ├── inspect_lora.py
│   ├── inspect_lora_values.py
│   ├── analyze_lora_update.py
│   ├── lora_training_demo.py
│   └── compare_finetuning.py
│
├── 10_quantization/
│   ├── quantization_demo.py
│   ├── int4_demo.py
│   └── qlora_concept_demo.py
│
└── 11_transformer_from_scratch/
```

The empty folders are intentional. They will be populated as the corresponding concepts are implemented.

---

# Part 9 — Transformer From Scratch

The next major stage is to remove the pre-built Transformer from the learning path.

Instead of:

```python
AutoModelForCausalLM.from_pretrained(...)
```

we will implement the core architecture ourselves using PyTorch.

The planned model:

```text
Tokenizer
   ↓
Embedding
   ↓
RoPE
   ↓
Multi-Head / Grouped Attention
   ↓
RMSNorm
   ↓
SwiGLU
   ↓
Transformer Block × N
   ↓
LM Head
   ↓
Loss
   ↓
Backpropagation
   ↓
Training
   ↓
Generation
```

The model will be deliberately small so that it can be trained on a CPU.

The purpose is **understanding, not competitive model performance**.

---

# Key Concepts Learned

By completing the current experiments, the following pipeline has been demonstrated:

```text
TEXT
 ↓
TOKENIZER
 ↓
TOKEN IDs
 ↓
EMBEDDINGS
 ↓
Q / K / V
 ↓
ROPE
 ↓
GQA
 ↓
MULTI-HEAD ATTENTION
 ↓
OUTPUT PROJECTION
 ↓
RESIDUAL
 ↓
RMSNORM
 ↓
SWIGLU MLP
 ↓
TRANSFORMER BLOCKS
 ↓
FINAL HIDDEN STATE
 ↓
LM HEAD
 ↓
LOGITS
 ↓
SOFTMAX
 ↓
NEXT TOKEN
```

And the training mechanism:

```text
PREDICTION
 ↓
LOSS
 ↓
BACKPROPAGATION
 ↓
GRADIENTS
 ↓
OPTIMIZER
 ↓
WEIGHT UPDATE
 ↓
LEARNING
```

Adaptation:

```text
Fine-tuning
    ↓
LoRA
    ↓
Quantization
    ↓
QLoRA
```

---

# Environment

The experiments were developed using:

```text
Python 3.11
PyTorch
Hugging Face Transformers
PEFT
Datasets
Accelerate
```

The experiments are designed to run locally and are intentionally small where practical.

The current learning environment is CPU-based.

---

# Philosophy

This repository follows a simple principle:

> **Don't just call the model. Understand the model.**

Instead of treating an LLM as a black box, each experiment exposes another layer of the computation.

The final objective is to be able to look at:

```text
text → tokens → tensors → attention → hidden states → logits → loss → gradients → weights
```

and understand what is happening at every stage.

---

# Status

## v0.1 — Completed

* Existing LLM inspection
* Tokenization and embeddings
* Q/K/V
* GQA
* Attention
* RoPE
* Attention output projection
* Residual connections
* RMSNorm
* MLP / SwiGLU
* LM head
* Logits and probabilities
* Loss
* Backpropagation
* Optimizer updates
* Training loop
* Fine-tuning
* LoRA
* Quantization
* QLoRA concepts
* Inference and sampling

## v0.2 — In Progress

* Repository reorganization
* Transformer implementation from scratch

## Future

* Train the tiny Transformer locally
* Inspect its learned weights
* Evaluate generation
* Experiment with different datasets
* Compare the scratch implementation with SmolLM2
* Explore more advanced training and evaluation techniques
