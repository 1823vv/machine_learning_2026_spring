## Question: One Gradient Descent Step by Hand in Logistic Regression

Consider one training sample:

$$
x=[1,2], \quad y=1
$$

and a logistic regression model with:

$$
W=\begin{bmatrix}0.5\\-0.5\end{bmatrix}, \quad b=0
$$

Use learning rate $\eta=0.1$ and assume:

$$
\sigma(-0.5)\approx 0.38
$$

1. Compute $z=xW+b$.
2. Compute $\hat{y}$.
3. Compute the error signal $\hat{y}-y$.
4. Compute the gradient with respect to $W$ for this one sample:

   $$
   \frac{\partial \ell}{\partial W}=x^{\mathsf{T}}(\hat{y}-y)
   $$

5. Compute the gradient with respect to $b$.
6. Update $W$ and $b$ using the leftarrow notation.
7. Explain in words why the first weight increases and the second weight becomes less negative after this update.
