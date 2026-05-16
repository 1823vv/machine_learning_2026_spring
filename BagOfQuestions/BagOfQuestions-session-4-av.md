## Question: Batch Backpropagation for Linear Regression

Treat linear regression as a one-layer neural network with identity activation. For a batch of $n$ examples,

$$
\hat{Y} = XW + \mathbf{1}b,
$$

where $X \in \mathbb{R}^{n \times d}$, $W \in \mathbb{R}^{d \times 1}$, $b \in \mathbb{R}^{1 \times 1}$, $Y \in \mathbb{R}^{n \times 1}$, and $\mathbf{1} \in \mathbb{R}^{n \times 1}$. The average mean squared error is

$$
\mathcal{L} = \frac{1}{n}\sum_{i=1}^n (\hat y^{(i)} - y^{(i)})^2.
$$

1. Write the shape of $\hat{Y}$.
2. Write the matrix formula for $\frac{\partial \mathcal{L}}{\partial \hat{Y}}$.
3. Since the activation is identity, what is $\Delta = \frac{\partial \mathcal{L}}{\partial Z}$?
4. Write the matrix formula for $\frac{\partial \mathcal{L}}{\partial W}$.
5. Write the matrix formula for $\frac{\partial \mathcal{L}}{\partial b}$.
6. Check the shapes of $\Delta$, $X^T\Delta$, and $\mathbf{1}^T\Delta$.
7. Explain why averaging by $n$ changes the gradient scale but not the direction.
8. Draw the batch computation graph and label every matrix shape.
