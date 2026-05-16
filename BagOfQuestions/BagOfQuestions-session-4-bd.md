## Question: Mini-Batch Backpropagation for Logistic Regression

Treat binary logistic regression as a sigmoid output layer. For a mini-batch of $B$ examples,

$$
Z = XW + \mathbf{1}b,
$$

$$
\hat{Y}=\sigma(Z),
$$

where $X \in \mathbb{R}^{B \times d}$, $W \in \mathbb{R}^{d \times 1}$, $b \in \mathbb{R}^{1 \times 1}$, and $Y \in \mathbb{R}^{B \times 1}$. The average binary cross-entropy loss is used.

1. Write the shape of $Z$ and $\hat{Y}$. Write the error-signal matrix $\Delta=\frac{\partial \mathcal{L}}{\partial Z}$ for average binary cross-entropy.
2. Write $\frac{\partial \mathcal{L}}{\partial W}$. Write $\frac{\partial \mathcal{L}}{\partial b}$.
3. Explain where the factor $\frac{1}{B}$ appears. Write the update rules for $W$ and $b$ using learning rate $\eta$.
4. Explain why the same formulas work for any mini-batch size $B$ as long as the shapes match. Draw a mini-batch computation graph with forward arrows and backward gradient arrows.
