## Question: Sparse Labels versus One-Hot Labels

For multiclass classification with $K$ classes, the target can be stored either as a sparse class index or as a one-hot vector.

For one example with correct class $c$, the sparse label stores only $c$. The one-hot label stores a vector $y \in \mathbb{R}^{1 \times K}$ where

$$
y_c = 1
$$

and

$$
y_k = 0 \quad \text{for} \quad k \neq c.
$$

1. What is the shape of a one-hot label for one example?
2. What is the shape of a one-hot label matrix for $n$ examples? What information is stored by a sparse class label?
3. Explain why sparse labels and one-hot labels can represent the same target. Why might sparse labels save memory when $K$ is large?
4. Explain why a sparse class index should not be interpreted as a continuous numerical target.
