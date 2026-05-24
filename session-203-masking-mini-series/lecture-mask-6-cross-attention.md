# Cross-Attention Masking

---

## 1. Core Question

Cross-attention connects two sequences:

* Query side: decoder
* Key/Value side: encoder

This creates a new masking problem:

> What constraints should be applied when attention flows across two different sequences?

Unlike self-attention, we cannot simply reuse one mask type. We must reason about **where invalid information can come from**.

---

## 2. Attention Structure

Cross-attention computes:

$$
\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

with:

* $Q \in \mathbb{R}^{T_d \times d_k}$ = query matrix from decoder
* $K \in \mathbb{R}^{T_e \times d_k}$ = key matrix from encoder
* $V \in \mathbb{R}^{T_e \times d_v}$ = value matrix from encoder

The output is:

$$
o_i = \sum_{j=1}^{T_e} \alpha_{ij} v_j
$$

So each decoder position $i$ aggregates information from encoder positions $j$.

---

## 3. What Can Go Wrong?

Even though cross-attention does not involve temporal ordering inside one sequence, it still has a fundamental issue:

> Not all encoder positions are valid.

In practice, encoder inputs are padded:

```
[The] [cat] [sat] [PAD] [PAD]
```

Without masking:

* Decoder attends to padding tokens
* Padding contributes meaningless vectors
* Attention distribution becomes distorted

---

## 4. Key Constraint in Cross-Attention

The only meaningful constraint is:

> Decoder must only attend to **valid encoder tokens**

There is no constraint across decoder positions here.

---

### Formal Validity Condition

For encoder position $j$:

$$
j \text{ is valid} \Rightarrow m_j = 1
$$

We require:

$$
\alpha_{ij} = 0 \quad \text{if } m_j = 0
$$

---

## 5. Mask Construction

We define a mask over encoder positions only:

$$
M_{ij} =
\begin{cases}
0 & m_j = 1 \\
-\infty & m_j = 0
\end{cases}
$$

Important properties:

* Independent of decoder index $i$
* Shared across all decoder queries
* Acts only on **keys/values side**

---

## 6. Masked Cross-Attention

Final formulation:

$$
\text{CrossAttention}(Q, K, V)=
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Where:

* $Q \in \mathbb{R}^{T_d \times d_k}$ = query matrix from decoder
* $K \in \mathbb{R}^{T_e \times d_k}$ = key matrix from encoder
* $V \in \mathbb{R}^{T_e \times d_v}$ = value matrix from encoder

After softmax:

* invalid encoder positions receive probability 0
* valid encoder positions are renormalized

---

## 7. Graph Interpretation

Cross-attention can be viewed as a **bipartite graph**:

* Left nodes: decoder positions
* Right nodes: encoder positions

Edges represent information flow from encoder to decoder.

---

### Without Mask

```
All decoder nodes → all encoder nodes
```

Every decoder position can access:

* real tokens
* padding tokens (invalid)

---

### With Mask

```
All decoder nodes → only valid encoder nodes
```

Padding nodes are completely removed from the graph.

---

## 8. Important Design Insight

Cross-attention masking is simpler than self-attention masking because:

* There is **no temporal structure in encoder-decoder interaction**
* There is **no need for causal constraints**
* The only structural issue is **data validity**

---

### Comparison

| Component              | Constraint Type      |
| ---------------------- | -------------------- |
| Encoder self-attention | Padding only         |
| Decoder self-attention | Causal + padding     |
| Cross-attention        | Encoder padding only |

---

## 9. Why This Works

The encoder produces a set of representations:

$$
h_1, h_2, \dots, h_{T_e}
$$

But not all positions are meaningful.

Padding positions:

* Do not correspond to real input tokens
* Carry no semantic content
* Should not influence decoding

Masking ensures:

> Cross-attention becomes a pure retrieval mechanism over valid memory entries
