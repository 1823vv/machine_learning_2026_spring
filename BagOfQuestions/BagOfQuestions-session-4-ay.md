## Question: One Backpropagation Step by Hand for Logistic Regression

Consider one binary logistic-regression example treated as a sigmoid neuron. Let

$$
x = \begin{bmatrix}1 & 2\end{bmatrix},
\quad
W = \begin{bmatrix}0.2 \\ -0.1\end{bmatrix},
\quad
b = \begin{bmatrix}0\end{bmatrix},
\quad
y=1.
$$

Use $z=xW+b$, $\hat y=\sigma(z)$, and binary cross-entropy loss. You may use $\sigma(0)=0.5$.

1. Compute $z$. Compute $\hat y$.
2. Compute the binary cross-entropy loss. Compute $\delta=\frac{\partial \ell}{\partial z}$.
3. Compute $\frac{\partial \ell}{\partial W}=x^T\delta$. Compute $\frac{\partial \ell}{\partial b}$.
4. With learning rate $\eta=0.1$, compute the updated $W$ and $b$. After the update, should the logit for this same example increase or decrease? Explain using the target label.
