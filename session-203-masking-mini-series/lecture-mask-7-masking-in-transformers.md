# Masking in Transformer Architectures

## 1. Motivation

So far, we studied masking as an isolated mechanism:

* Causal mask controls temporal access.
* Padding mask controls token validity.
* Combined mask enforces both constraints.
* Cross-attention mask filters encoder-side padding.


Now we place masking inside the **Transformer architecture**.

> [!INFO]
> The key question: where exactly are masks applied in different Transformer models?

## 2. Three Transformer Variants

Modern models fall into three categories:

1. **Encoder-only models** (e.g., BERT)
2. **Decoder-only models** (e.g., GPT)
3. **Encoder-decoder models** (original Transformer)

Each uses masking differently.

## 3. Encoder-Only Models (BERT)

### Structure

* Input: full sequence available
* Task: representation learning (classification, embedding)

### Attention-Level Masking

Encoder self-attention uses:

* **Padding mask only**
* No causal constraint

### Why No Causal Mask?

Because:

* The model is not generating tokens
* All tokens are available simultaneously

So attention is:

```
i → 1, 2, ..., T   (except PAD)
```

### Masked Language Model Pre-training

Beyond attention masking, BERT employs a distinct masking strategy at the **input level** during pre-training: the **Masked Language Model (MLM)** objective.

> [!INFO]
> MLM masking corrupts the input sequence by replacing tokens, forcing the model to reconstruct the original text from bidirectional context. This is fundamentally different from attention masking, which only controls which positions participate in attention computation.

#### The Masking Procedure

During pre-training, BERT randomly selects approximately 15% of the input tokens. For each selected token:

* 80% of the time: replace with the special `[MASK]` token
* 10% of the time: replace with a random token from the vocabulary
* 10% of the time: keep the original token unchanged

The model then predicts the original token at each masked position using bidirectional context.

#### Analogy to Cloze Test

The MLM objective is conceptually identical to the **Cloze Test** in language assessment:

* **Cloze Test**: Words are deleted from a passage; the reader must infer the missing words from surrounding context.
* **BERT MLM**: Tokens are masked in a sequence; the model must predict the original tokens from bidirectional context.

This is why the MLM objective is often described as forcing the model to solve "fill-in-the-blank" problems at scale. The key difference from standard cloze tests is that BERT's masking is randomly applied and the model must handle the `[MASK]` symbol, random substitutions, and unchanged tokens to prevent overfitting to any single masking pattern.

> [!WARNING]
> Do not confuse **MLM input masking** with **attention masking**. MLM modifies the input tokens themselves. Attention masking (padding mask) only controls which positions participate in the self-attention computation.

### Summary

* Attention mask used: **padding mask**
* Pre-training mask: **[MASK]** token replacement with bidirectional prediction
* No restriction on temporal order
* Fully bidirectional attention

## 4. Decoder-Only Models (GPT)

### Structure

* Input: sequence of tokens
* Task: autoregressive generation

### Masking Behavior

Decoder self-attention uses:

* **Causal mask**
* **Padding mask (if batching)**

### Attention Pattern

```
1 → 1
2 → 1, 2
3 → 1, 2, 3
...
```

### Why Both Masks?

* Causal mask → prevents future leakage
* Padding mask → removes invalid tokens

## 5. Encoder-Decoder Models (Original Transformer)

### Structure

The model consists of:

1. Encoder stack
2. Decoder stack

### Three Attention Types

Each layer involves different masking:

#### 5.1 Encoder Self-Attention

* Q, K, V from encoder
* Mask: **padding mask**

```
i → all valid encoder tokens
```

#### 5.2 Decoder Self-Attention

* Q, K, V from decoder
* Mask: **causal + padding**

```
i → past decoder tokens only
```

#### 5.3 Cross-Attention

* Q from decoder
* K, V from encoder
* Mask: **encoder padding mask only**

```
decoder i → all valid encoder tokens
```

## 6. Unified View

We can summarize all masking behaviors:

| Attention Type    | Q Source | K/V Source | Mask             |
| ----------------- | -------- | ---------- | ---------------- |
| Encoder self-attn | Encoder  | Encoder    | Padding          |
| Decoder self-attn | Decoder  | Decoder    | Causal + Padding |
| Cross-attn        | Decoder  | Encoder    | Encoder Padding  |

## 7. Mathematical View

All attention layers follow the same formula:

$$
\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Where:

* $Q \in \mathbb{R}^{T \times d_k}$ = query matrix
* $K \in \mathbb{R}^{T \times d_k}$ = key matrix
* $V \in \mathbb{R}^{T \times d_v}$ = value matrix

The only difference is:

> [!INFO]
> How we construct $M$

### Examples

* Encoder:
  $$
  M = M_{\text{padding}}
  $$

* Decoder self-attention:
  $$
  M = M_{\text{causal}} + M_{\text{padding}}
  $$

* Cross-attention:
  $$
  M = M_{\text{encoder padding}}
  $$