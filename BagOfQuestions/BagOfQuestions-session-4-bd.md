## Question: Mini-Batch Backpropagation for Logistic Regression

Treat binary logistic regression as a sigmoid output layer. For a mini-batch of $B$ examples,

$$
Z = XW + \mathbf{1}b,
$$

$$
\hat{Y}=\sigma(Z),
$$

where $X \in \mathbb{R}^{B \times d}$, $W \in \mathbb{R}^{d \times 1}$, $b \in \mathbb{R}^{1 \times 1}$, and $Y \in \mathbb{R}^{B \times 1}$. The average binary cross-entropy loss is used.

1. Write the shape of $Z$ and $\hat{Y}$.
2. Write the error-signal matrix $\Delta=\frac{\partial \mathcal{L}}{\partial Z}$ for average binary cross-entropy.
3. Write $\frac{\partial \mathcal{L}}{\partial W}$.
4. Write $\frac{\partial \mathcal{L}}{\partial b}$.
5. Explain where the factor $\frac{1}{B}$ appears.
6. Write the update rules for $W$ and $b$ using learning rate $\eta$.
7. Explain why the same formulas work for any mini-batch size $B$ as long as the shapes match.
8. Draw a mini-batch computation graph with forward arrows and backward gradient arrows.
