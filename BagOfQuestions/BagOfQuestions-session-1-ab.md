## Question: One Gradient Descent Step for Simple Linear Regression

Consider a simple linear regression model:

$$
\hat{y}=wx+b
$$

with one training sample:

$$
x=2, \quad y=5, \quad w=1, \quad b=0
$$

Use the one-sample squared error:

$$
\ell=(\hat{y}-y)^2
$$

and learning rate $\eta=0.1$.

1. Compute $\hat{y}$.
2. Compute the error $\hat{y}-y$.
3. Compute $\frac{\partial \ell}{\partial w}=2(\hat{y}-y)x$.
4. Compute $\frac{\partial \ell}{\partial b}=2(\hat{y}-y)$.
5. Update $w$ and $b$ using leftarrow notation.
6. After the update, does the prediction for $x=2$ move closer to 5? Check numerically.
7. Draw the loss as a bowl-shaped curve and show a gradient descent step moving downhill.

## Question: Learning Rate Intuition

1. What can happen if the learning rate is too small?
2. What can happen if the learning rate is too large?
3. Draw three optimization paths: too slow, good, and unstable.
