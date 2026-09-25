import torch
import math

# 3 tokens, each represented by 4 numbers
X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0], # token A
    [0.0, 1.0, 0.0, 1.0], # token B
    [1.0, 1.0, 0.0, 0.0], # token C
])

print("Input X:")
print(X)

print("\nShape:")
print(X.shape)

# Weight matrices
Wq = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
])

Wk = Wq.clone()

Wv = torch.tensor([
    [1.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 0.0],
])

Q = X @ Wq
K = X @ Wk
V = X @ Wv

print("\nQ:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)

scores = Q @ K.T

print("\nAttention scores:")
print(scores)

print("\nScore shape:")
print(scores.shape)

attention_weights = torch.softmax(scores, dim=-1)

print("\nAttention weights:")
print(attention_weights)

print("\nRow sums:")
print(attention_weights.sum(dim=-1))

attention_output = attention_weights @ V

print("\nAttention output:")
print(attention_output)

print("\nOutput shape:")
print(attention_output.shape)

d = Q.shape[-1]

scaled_scores = scores / math.sqrt(d)

scaled_weights = torch.softmax(scaled_scores, dim=-1)

print("\nScaled scores:")
print(scaled_scores)

print("\nScaled attention weights:")
print(scaled_weights)

mask = torch.tril(torch.ones(3, 3))

print("\nCausal mask:")
print(mask)

masked_scores = scaled_scores.masked_fill(mask == 0, float("-inf"))

masked_weights = torch.softmax(masked_scores, dim=-1)

print("\nMasked scores:")
print(masked_scores)

print("\nMasked attention weights:")
print(masked_weights)

print("\nRow sums:")
print(masked_weights.sum(dim=-1))

# Split X into 2 attention heads
head1 = X[:, 0:2]
head2 = X[:, 2:4]

print("\nHead 1:")
print(head1)

print("\nHead 2:")
print(head2)

print("\nHead shapes:")
print("Head 1:", head1.shape)
print("Head 2:", head2.shape)

# Each head uses its own Q, K, V
Wq1 = torch.eye(2)
Wk1 = torch.eye(2)
Wv1 = torch.eye(2)

Wq2 = torch.eye(2)
Wk2 = torch.eye(2)
Wv2 = torch.eye(2)

Q1 = head1 @ Wq1
K1 = head1 @ Wk1
V1 = head1 @ Wv1

Q2 = head2 @ Wq2
K2 = head2 @ Wk2
V2 = head2 @ Wv2

print("\nHead 1 Q:")
print(Q1)

print("\nHead 2 Q:")
print(Q2)

scores1 = Q1 @ K1.T

d1 = Q1.shape[-1]

scaled_scores1 = scores1 / math.sqrt(d1)

weights1 = torch.softmax(scaled_scores1, dim=-1)

output1 = weights1 @ V1

print("\nHead 1 scores:")
print(scores1)

print("\nHead 1 attention weights:")
print(weights1)

print("\nHead 1 output:")
print(output1)

scores2 = Q2 @ K2.T

d2 = Q2.shape[-1]

scaled_scores2 = scores2 / math.sqrt(d2)

weights2 = torch.softmax(scaled_scores2, dim=-1)

output2 = weights2 @ V2

print("\nHead 2 scores:")
print(scores2)

print("\nHead 2 attention weights:")
print(weights2)

print("\nHead 2 output:")
print(output2)

multi_head_output = torch.cat([output1, output2], dim=-1)

print("\nMulti-head output:")
print(multi_head_output)

print("\nMulti-head output shape:")
print(multi_head_output.shape)