## Question: Multi-Output Linear Layer Backpropagation

Consider a linear layer with row-vector input and two outputs:

$$
Z = XW + \mathbf{1}b,
$$

where $X \in \mathbb{R}^{n \times d}$, $W \in \mathbb{R}^{d \times 2}$, $b \in \mathbb{R}^{1 \times 2}$, and $Z \in \mathbb{R}^{n \times 2}$. Suppose a later loss calculation gives

$$
\Delta = \frac{\partial \mathcal{L}}{\partial Z} \in \mathbb{R}^{n \times 2}.
$$

1. Write the formula for $\frac{\partial \mathcal{L}}{\partial W}$.
2. Write the formula for $\frac{\partial \mathcal{L}}{\partial b}$. Write the formula for $\frac{\partial \mathcal{L}}{\partial X}$.
3. Check the shape of each gradient. Explain why $\frac{\partial \mathcal{L}}{\partial X}$ is needed when this layer is not the first layer.
4. Explain how the one-output linear-regression case is recovered when the output dimension is 1. Draw the forward and backward arrows for this layer.
