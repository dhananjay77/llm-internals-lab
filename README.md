\# LLM Training \& Internals Lab



A hands-on local project for understanding how Large Language Models work — from tokenization and embeddings through Transformer attention, continued pretraining, and LoRA fine-tuning.



The goal is \*\*learning by inspecting and modifying real model tensors\*\*, rather than treating an LLM as a black box.



\## Model



This project currently uses:



\* \*\*Model:\*\* SmolLM2-135M

\* \*\*Architecture:\*\* Llama-style causal Transformer

\* \*\*Parameters:\*\* \~135M

\* \*\*Hidden size:\*\* 576

\* \*\*Transformer layers:\*\* 30

\* \*\*Attention heads:\*\* 9

\* \*\*Key/Value heads:\*\* 3

\* \*\*Head dimension:\*\* 64

\* \*\*Vocabulary:\*\* 49,152 tokens

\* \*\*Precision:\*\* BF16

\* \*\*Runtime:\*\* PyTorch / CPU



The model is intentionally small so the experiments can run locally.



\---



\## Learning Journey



The project follows the LLM pipeline:



```text

Text

&#x20; ↓

Tokenizer

&#x20; ↓

Token IDs

&#x20; ↓

Embeddings

&#x20; ↓

Transformer Layers

&#x20; ↓

Attention

&#x20; ↓

MLP / Feed Forward

&#x20; ↓

Next-token prediction

&#x20; ↓

Loss

&#x20; ↓

Backpropagation

&#x20; ↓

Parameter updates

```



Training techniques:



```text

Domain Corpus

&#x20;    ↓

Continued Pretraining

&#x20;    ↓

Domain-adapted Model

&#x20;    ↓

LoRA Fine-tuning

&#x20;    ↓

LoRA Adapter

```



\---



\# Experiments Completed



\## 1. Tokenization



Example:



```text

A Value Object is defined by its

```



is converted into tokens such as:



```text

A

ĠValue

ĠObject

Ġis

Ġdefined

Ġby

Ġits

```



Token IDs are vocabulary indices. They are not semantic meanings.



\---



\## 2. Embeddings



Token IDs are mapped to learned vectors.



For this model:



```text

Token

&#x20; ↓

Embedding lookup

&#x20; ↓

576-dimensional vector

```



These vectors become the initial representations processed by the Transformer.



\---



\## 3. Q / K / V



For every Transformer attention layer, the hidden representation is projected into:



```text

Q = Query

K = Key

V = Value

```



The fundamental attention equation is:



```text

Attention(Q,K,V)

=

softmax(QKᵀ / √d) V

```



The project calculates these tensors directly from the model.



\---



\## 4. Grouped Query Attention



The model uses:



```text

Query heads:       9

Key/Value heads:   3

Head dimension:    64

```



Therefore:



```text

Q = 9 × 64 = 576

K = 3 × 64 = 192

V = 3 × 64 = 192

```



The 3 K/V heads are repeated to serve the 9 Query heads.



This is \*\*Grouped-Query Attention (GQA)\*\*.



\---



\## 5. Multi-Head Attention



For the sentence:



```text

A Value Object is defined by its

```



the model produces:



```text

\[batch, heads, sequence, sequence]



\[1, 9, 7, 7]

```



Therefore there are:



```text

9 attention matrices

```



for the 7-token sequence.



Different heads produce different attention distributions.



\---



\## 6. Causal Attention



Because this is a causal language model, a token cannot attend to future tokens.



For example:



```text

A

Value → A, Value

Object → A, Value, Object

is → A, Value, Object, is

...

```



The resulting attention matrix is lower triangular.



\---



\## 7. Attention Weights



For Head 0, the token `its` produced approximately:



```text

A          0.0659

Value      0.1988

Object     0.2112

is         0.2027

defined    0.1107

by         0.0377

its        0.1730

```



These values sum to approximately 1 because they are produced by softmax.



They represent how the attention calculation distributes weight across the available tokens for that head and position.



\---



\## 8. V-Weighted Attention Output



The attention weights are applied to the corresponding Value vectors:



```text

0.0659 × V(A)

\+ 0.1988 × V(Value)

\+ 0.2112 × V(Object)

\+ ...

\+ 0.1730 × V(its)

```



This produces a:



```text

64-dimensional vector

```



for Head 0 and the token `its`.



This demonstrates the complete core attention operation numerically.



\---



\# Continued Pretraining



A small domain corpus covering software engineering concepts was used for continued pretraining.



Topics include:



\* Domain-Driven Design

\* Entities

\* Value Objects

\* Aggregates

\* Bounded Contexts

\* Event-driven architecture

\* Microservices

\* Kubernetes

\* Containers

\* Observability

\* Distributed systems



The objective is next-token prediction on domain text.



```text

Generic Model

&#x20;    ↓

Domain Corpus

&#x20;    ↓

Continued Pretraining

&#x20;    ↓

Domain-adapted Model

```



Because the dataset is intentionally tiny, the resulting model should be considered an educational demonstration rather than a useful domain expert.



\---



\# LoRA Fine-tuning



A small instruction/response dataset was used for LoRA fine-tuning.



The adapter targets:



```text

q\_proj

v\_proj

```



with:



```text

rank (r) = 8

alpha = 16

dropout = 0.05

```



The LoRA update follows:



```text

W' = W + ΔW



ΔW = B × A

```



The original model weights remain frozen during LoRA training.



For this model:



```text

Total parameters       ≈ 134.98M

LoRA parameters        ≈ 460.8K

Trainable fraction     ≈ 0.34%

```



This demonstrates why parameter-efficient fine-tuning can require far fewer trainable parameters than full fine-tuning.



\---



\# Repository Experiments



Current scripts:



```text

attention\_demo.py

attention\_v\_demo.py

attention\_heads.py

analyze\_lora\_update.py

compare\_q.py

compare\_finetuning.py

```



Additional training scripts and datasets are also included in the repository.



The repository will be reorganized after the learning experiments are complete.



\---



\# Current Understanding



At this checkpoint, the following concepts have been explored hands-on:



\* Tokenization

\* Token IDs

\* Embeddings

\* Transformer architecture

\* Query / Key / Value

\* Attention scores

\* Scaling

\* Causal masking

\* Softmax

\* Multi-head attention

\* Grouped-Query Attention

\* Value aggregation

\* Continued pretraining

\* Fine-tuning

\* LoRA

\* Low-rank weight updates

\* Parameter-efficient training



\---



\# Next Experiments



The next stage will complete the Transformer forward pass:



```text

RoPE

&#x20;↓

Q/K positional transformation

&#x20;↓

Attention

&#x20;↓

Output projection

&#x20;↓

Residual connection

&#x20;↓

LayerNorm

&#x20;↓

MLP / Feed Forward Network

&#x20;↓

Residual connection

```



Then the project will investigate:



```text

Transformer layers

&#x20;↓

LM Head

&#x20;↓

Logits

&#x20;↓

Softmax

&#x20;↓

Next-token probabilities

&#x20;↓

Loss

&#x20;↓

Backpropagation

&#x20;↓

Gradients

&#x20;↓

Parameter updates

```



Finally:



```text

LoRA

&#x20;↓

Quantization

&#x20;↓

QLoRA

&#x20;↓

Local inference

```



\---



\# Philosophy



The purpose of this project is not to build the largest model.



It is to answer:



> \*\*What actually happens inside an LLM when text enters the model and when the model learns?\*\*



Every major concept should eventually have a small, executable experiment demonstrating it with real tensors.



\---



\## Status



\*\*Version:\*\* v0.1



\*\*Focus:\*\* LLM training fundamentals + Transformer attention internals



\*\*Next milestone:\*\* Complete one Transformer block end-to-end.



