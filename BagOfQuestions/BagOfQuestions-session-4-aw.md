## Question: One Backpropagation Step by Hand for Linear Regression

Consider one training example for a one-layer linear-regression neural network. Let

$$
x = \begin{bmatrix}2 & -1\end{bmatrix},
\quad
W = \begin{bmatrix}0.5 \\ -0.25\end{bmatrix},
\quad
b = \begin{bmatrix}0.1\end{bmatrix},
\quad
y = \begin{bmatrix}1.0\end{bmatrix}.
$$

The forward equations are $z=xW+b$ and $\hat y=z$. The loss is $\ell=(\hat y-y)^2$.

1. Compute $z$. Compute $\hat y$.
2. Compute $\ell$. Compute $\frac{\partial \ell}{\partial \hat y}$.
3. Compute $\delta=\frac{\partial \ell}{\partial z}$. Compute $\frac{\partial \ell}{\partial W}=x^T\delta$.
4. Compute $\frac{\partial \ell}{\partial b}$. With learning rate $\eta=0.1$, compute the updated $W$ and $b$. Draw the scalar computation graph and place each gradient beside the corresponding edge.
