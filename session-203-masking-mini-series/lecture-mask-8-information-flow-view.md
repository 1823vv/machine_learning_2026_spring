# Masking as Information Flow Control

---

## 1. Motivation

So far, we have studied three types of masking:

* Causal mask → prevents access to future tokens
* Padding mask → removes invalid tokens
* Cross-attention mask → filters encoder information

Each was introduced separately, tied to a specific problem.

Now we take a step back and ask:

> Is there a single principle that explains all masking strategies?

---

## 2. From Computation to Information Flow

Recall the attention formula:

$$
\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Where:

* $Q \in \mathbb{R}^{T \times d_k}$ = query matrix
* $K \in \mathbb{R}^{T \times d_k}$ = key matrix
* $V \in \mathbb{R}^{T \times d_v}$ = value matrix

At a mechanical level:

* $QK^T$ computes similarities
* Softmax produces weights
* Weighted sum aggregates values

---

### A More Conceptual View

For a given position $i$:

$$
o_i = \sum_{j} \alpha_{ij} v_j
$$

This can be interpreted as:

> Information flows from position $j$ to position $i$ with strength $\alpha_{ij}$

---

### Key Insight

> Attention defines a **learned information flow network**

---

## 3. Masking Controls Information Flow

Masking modifies this flow:

* If $M_{ij} = 0$ → information is allowed to flow
* If $M_{ij} = -\infty$ → information is blocked

After softmax:

$$
\alpha_{ij} = 0
$$

So no information passes from $j$ to $i$.

---

### Unified Interpretation

> Masking = controlling which information paths are allowed

---

## 4. Three Types of Constraints

All masking strategies correspond to **different constraints on information flow**.

---

### 4.1 Temporal Constraint (Causal Mask)

Constraint:

$$
\text{Information cannot flow from future to past}
$$

Effect:

* Blocks edges where $j > i$
* Enforces autoregressive structure

---

### 4.2 Validity Constraint (Padding Mask)

Constraint:

$$
\text{Information cannot flow from non-existent tokens}
$$

Effect:

* Blocks edges pointing to padding positions
* Removes meaningless contributions

---

### 4.3 Source Constraint (Cross-Attention)

Constraint:

$$
\text{Information flows only from encoder to decoder}
$$

Effect:

* Allows decoder to access encoder
* Filters out encoder padding

---

## 5. Graph Perspective Revisited

From the graph viewpoint:

* Nodes = tokens
* Edges = possible information flow

---

### Without Masking

* Fully connected graph
* All information flows freely

---

### With Masking

* Some edges are removed
* Only valid paths remain

---

### Interpretation

> The Transformer is not just computing attention—it is **designing a structured information flow graph**

---

## 6. Architecture as Flow Design

We can reinterpret Transformer architectures:

---

### Encoder (BERT-style)

* Bidirectional flow
* No temporal restriction
* Only removes invalid nodes

---

### Decoder (GPT-style)

* Directed flow (past → future)
* Enforces strict temporal ordering

---

### Encoder-Decoder (Original Transformer)

* Two-stage flow:

  1. Encoder builds representations
  2. Decoder selectively reads them

---

## 7. Why This Perspective Matters

This unified view helps explain:

---

### 7.1 Why Masking Is Necessary

Without constraints:

* The model learns invalid dependencies
* Information flows in unrealistic ways

---

### 7.2 Why Different Models Use Different Masks

Because they impose different **information flow structures**:

* GPT → causal flow
* BERT → bidirectional flow
* Encoder-decoder → hierarchical flow

---

### 7.3 Why Masking Is Not a Trick

Masking is often presented as an implementation detail.

But in reality:

> Masking defines the **structure of the model**

---

## 8. Practical Implications

---

### Model Design

When designing a model, you are choosing:

* What information is accessible
* What dependencies are allowed

---

### Training Stability

Proper masking ensures:

* No leakage
* Clean gradients
* Meaningful representations

---

### Generalization

Models generalize better when:

* Training constraints match inference constraints
* Information flow is realistic

---

## 9. Final Summary

All masking strategies can be unified as:

$$
\text{Masking} = \text{Constraining Information Flow}
$$

Specifically:

* Causal mask → controls **when** information flows
* Padding mask → controls **whether** information exists
* Cross-attention mask → controls **where** information comes from

---

## 10. Key Takeaways

* Attention defines a **network of information flow**
* Masking removes invalid or undesired connections
* Different masks correspond to different **structural constraints**
* Transformer architectures are defined by how they **control information flow**
