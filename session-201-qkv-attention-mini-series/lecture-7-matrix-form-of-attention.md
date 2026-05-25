# Self-Attention in Matrix Form

![](./img-2/transformer-explained.jpg)

---

## 1. From Token-Level to Matrix Computation (again)

We have already seen the matrix form of attention in earlier lectures when introducing Q, K, V and scaling. However, those discussions focused on *how the computation works internally*, rather than how it is organized across a full sequence.

In our previous lecture, we described attention at the **token level**:

$$
z_i = \sum_{j=0}^{n-1} \alpha_{ij} v_j
$$

with:

$$
\alpha_{ij}=
\frac{\exp\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)}
{\sum_{l=0}^{n-1} \exp\left(\frac{q_i \cdot k_l}{\sqrt{d_k}}\right)}
$$

This makes the computation of a single token explicit: one query interacts with all keys to produce one output.

The matrix form does not introduce a new mechanism. It simply reorganizes these per-token computations so that all queries are processed in parallel over the same set of key-value interactions.


---

## 2. Input and Linear Projections

Let the input sequence be:

$$
X \in \mathbb{R}^{n \times d_{\text{model}}}
$$

We project it into three representation spaces:

$$
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
$$

where:

* $Q, K \in \mathbb{R}^{n \times d_k}$
* $V \in \mathbb{R}^{n \times d_v}$

Each row corresponds to a token embedding in a different role space.

---

## 3. Score Matrix

![](./img-1/s9.jpg)

We compute all pairwise similarities:

$$
S = Q K^T
$$

where:

$$
S \in \mathbb{R}^{n \times n}, \quad s_{ij} = q_i \cdot k_j
$$

Each row $i$ contains the compatibility between query $q_i$ and all keys.

### Interpretation of $S$

$$
\begin{array}{c|ccccc}
& k_0 & k_1 & k_2 & \dots & k_{n-1} \\
\hline
q_0 & s_{00} & s_{01} & s_{02} & \dots & s_{0,n-1} \\
q_1 & s_{10} & s_{11} & s_{12} & \dots & s_{1,n-1} \\
q_2 & s_{20} & s_{21} & s_{22} & \dots & s_{2,n-1} \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
q_{n-1} & s_{n-1,0} & s_{n-1,1} & s_{n-1,2} & \dots & s_{n-1,n-1}
\end{array}
$$


* $s_{ij} > 0$: strong compatibility
* $s_{ij} < 0$: weak or opposing compatibility
* $s_{ij} = 0$: neutral relation

---

## 4. Normalization

We apply row-wise softmax on the scaled similarity scores $S$:

$$
A = \text{softmax}(\frac{S}{\sqrt{d_k}})
$$

so that:

$$
A_{ij} = \alpha_{ij}, \quad \sum_{j=0}^{n-1} A_{ij} = 1
$$


$$
\begin{array}{c|ccccc}
& 0 & 1 & 2 & \dots & n-1 \\
\hline
0 & \alpha_{00} & \alpha_{01} & \alpha_{02} & \dots & \alpha_{0,n-1} \\
1 & \alpha_{10} & \alpha_{11} & \alpha_{12} & \dots & \alpha_{1,n-1} \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
n-1 & \alpha_{n-1,0} & \alpha_{n-1,1} & \alpha_{n-1,2} & \dots & \alpha_{n-1,n-1}
\end{array}
$$


Each row defines a probability distribution over all tokens.

---

## 5. Weighted Aggregation

We compute the output as:

$$
Z = A V
$$

where:

$$
Z \in \mathbb{R}^{n \times d_v}
$$

Each output row is:

$$
z_i = \sum_{j=0}^{n-1} A_{ij} v_j
$$

So each token becomes a mixture of all value vectors.

---

## 6. Final Matrix Form

![](./img-3/selfattentionhs.jpg)

The full self-attention operation is:


$$
\boxed{\text{Attention}(X)=
\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V}
$$

or, more generally:

$$
\boxed{\text{Attention}(Q, K, V)=
\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V}
$$


---


## 7. The nanochat Implementation

Let's examine how Q, K, V are implemented in nanochat's `CausalSelfAttention`:

```python
class CausalSelfAttention(nn.Module):
        # ...
        # ...
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # B (Batch size): the number of sequences processed in parallel
        # T (Time / sequence length / n): the number of tokens in each sequence
        # C (Channel / embedding dimension / d_model): the feature size of each token
        B, T, C = x.size()
        
        # Project input to Q, K, V
        # Shape: (Batch, Time, Heads, HeadDim)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        # ...
        # ...
        return y
```

* `c_q`, `c_k`, `c_v` are the learned projection matrices ($W_Q$, $W_K$, $W_V$)
* `.view()` is PyTorch’s **reshape operation**.
    * It **returns a new tensor sharing the same underlying data** but with a different shape.
    * The **total number of elements must remain the same**, otherwise an error occurs.
    * `.view(B, T, H, D)` **reinterprets the original `(B, T, C)` tensor** as multiple heads (`H`) each with `D = head_dim` features, without changing the underlying data.
* head_dim = $d_k$ = $d_q$ (usually we don't see $d_q$ because it's always equal to $d_k$; recap: $\text{Score} = Q K^T$)


