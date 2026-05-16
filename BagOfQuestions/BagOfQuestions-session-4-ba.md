## Question: Shape Checking for Backpropagation in One-Layer Models

A one-layer model uses the row-vector batch convention

$$
Z = XW + \mathbf{1}b,
$$

where $X \in \mathbb{R}^{n \times d}$, $W \in \mathbb{R}^{d \times 1}$, $b \in \mathbb{R}^{1 \times 1}$, and $Z \in \mathbb{R}^{n \times 1}$. Let

$$
\Delta = \frac{\partial \mathcal{L}}{\partial Z} \in \mathbb{R}^{n \times 1}.
$$

1. What is the shape of $X^T$? What is the shape of $X^T\Delta$?
2. Why does $X^T\Delta$ match the shape of $W$? What is the shape of $\mathbf{1}^T\Delta$?
3. Why does $\mathbf{1}^T\Delta$ match the shape of $b$? Explain why $\Delta X$ is not the correct matrix multiplication for the weight gradient.
4. Draw the correct shape multiplication for $\frac{\partial \mathcal{L}}{\partial W}$. Draw the correct shape multiplication for $\frac{\partial \mathcal{L}}{\partial b}$.
