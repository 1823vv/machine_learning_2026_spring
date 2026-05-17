## Question: One Gradient Descent Step by Hand in Logistic Regression

Consider one training sample

$$
x=\begin{bmatrix}1 & 2\end{bmatrix}, \quad y=1
$$

and a logistic regression model with

$$
W=\begin{bmatrix}0.5\\-0.5\end{bmatrix}, \quad b=0.
$$

Use learning rate $\eta=0.1$ and assume $\sigma(-0.5)\approx 0.38$.

1. Compute $z=xW+b$, $\hat{y}$, and the error signal $\hat{y}-y$.
2. Compute the gradient with respect to $W$ for this one sample: $\frac{\partial \ell}{\partial W}=x^T(\hat{y}-y)$.
3. Compute the gradient with respect to $b$.
4. Update $W$ and $b$.