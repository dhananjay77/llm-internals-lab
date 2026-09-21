# LLM Internals Lab

A hands-on learning project to understand how Large Language Models work internally — from tokenization and embeddings to attention, training, LoRA, and eventually the complete Transformer learning loop.

## Model

**HuggingFaceTB/SmolLM2-135M**

Local CPU-based experiments using:

* Python 3.11.9
* PyTorch 2.14.0+cpu
* Transformers 5.17.0
* PEFT 0.21.0
* TRL 1.13.0

The model contains approximately **135 million parameters**.

---

## Learning Journey

```text
Text
  |
  v
Tokenizer
  |
  v
Token IDs
  |
  v
Embeddings
  |
  v
Transformer
  |
  +--> Q / K / V
  |
  +--> GQA
  |
  +--> Multi-Head Attention
  |
  +--> Feed Forward Network
  |
  +--> Residual + LayerNorm
  |
  v
Next Token Probabilities
  |
  v
Loss
  |
  v
Backpropagation
  |
  v
Parameter Updates
```

---

## 1. Tokenization

Text is converted into tokens and then into integer token IDs.

Example:

```text
"A Value Object is defined by its"
```

becomes approximately:

```text
A
Value
Object
is
defined
by
its
```

The token IDs are vocabulary indices. They do not directly represent the meaning of the words.

---

## 2. Embeddings

Token IDs are mapped to dense vectors through the model's embedding matrix.

Conceptually:

```text
Token ID
   |
   v
Embedding lookup
   |
   v
Vector
```

For SmolLM2-135M, the hidden representation size is:

```text
576 dimensions
```

Therefore each token becomes a 576-dimensional vector before entering the Transformer.

---

## 3. Query, Key and Value

The embedding/hidden representation is projected into three representations:

```text
X
 |
 +--> Q (Query)
 |
 +--> K (Key)
 |
 +--> V (Value)
```

The core attention equation is:

```text
Attention(Q, K, V)
    = softmax(QK^T / sqrt(d)) V
```

Where:

* **Q** = Query
* **K** = Key
* **V** = Value
* **d** = attention head dimension

---

## 4. Grouped Query Attention (GQA)

SmolLM2-135M uses Grouped Query Attention.

Its configuration is:

```text
Hidden size       = 576
Query heads       = 9
Key/Value heads   = 3
Head dimension    = 64
```

Therefore:

```text
Q = 9 x 64 = 576
K = 3 x 64 = 192
V = 3 x 64 = 192
```

The Key and Value heads are shared across groups of Query heads.

For our example:

```text
Q
[1, 9, 7, 64]

K
[1, 3, 7, 64]

V
[1, 3, 7, 64]
```

K and V are repeated to match the nine Query heads:

```text
Q
[1, 9, 7, 64]

K
[1, 9, 7, 64]

V
[1, 9, 7, 64]
```

The resulting attention matrix is:

```text
[1, 9, 7, 7]
```

That means:

```text
batch = 1
heads = 9
query positions = 7
key positions = 7
```

---

## 5. Causal Attention

An autoregressive LLM cannot look at future tokens.

For:

```text
A Value Object is defined by its
```

the token `Object` can attend to:

```text
A
Value
Object
```

but not to:

```text
is
defined
by
its
```

The causal mask prevents attention to future positions.

---

## 6. Multi-Head Attention

Each of the nine attention heads can produce a different attention distribution.

For the final token `its`, one experiment produced:

```text
HEAD 0
A          0.0659
Value      0.1988
Object     0.2112
is         0.2027
defined    0.1107
by         0.0377
its        0.1730
```

Another head produced a different distribution:

```text
HEAD 8
A          0.1305
Value      0.0062
Object     0.0164
is         0.5785
defined    0.0227
by         0.1961
its        0.0497
```

The distributions are different because each attention head has its own learned projections.

These numbers should not be interpreted as fixed semantic labels from a single example. Understanding what individual heads specialize in requires experiments across many inputs.

---

## 7. Attention Weights

For one head, the attention weights for `its` were:

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

---

## 8. V-Weighted Attention Output

The attention weights are applied to the corresponding Value vectors.

Conceptually:

```text
0.0659 * V(A)
+ 0.1988 * V(Value)
+ 0.2112 * V(Object)
+ 0.2027 * V(is)
+ 0.1107 * V(defined)
+ 0.0377 * V(by)
+ 0.1730 * V(its)
```

The result is a single 64-dimensional vector for that attention head and token position.

This demonstrates the second half of the attention equation:

```text
softmax(QK^T / sqrt(d)) V
```

The softmax produces the weights, and those weights determine how the Value vectors are combined.

---

## 9. Continued Pretraining

A small domain corpus was created containing software engineering concepts such as:

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

The model was then continued-pretrained using the next-token prediction objective.

Conceptually:

```text
Domain Text
    |
    v
Tokenizer
    |
    v
Tran
```
