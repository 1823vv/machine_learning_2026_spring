## Question: One Gradient Descent Step for Simple Linear Regression

Consider a simple linear regression model:

$$
\hat{y}=wx+b
$$

with one training sample:

$$
x=2, \quad y=5, \quad w=1, \quad b=0.
$$

Use the one-sample squared error

$$
\ell=(\hat{y}-y)^2
$$

and learning rate $\eta=0.1$.

1. Compute $\hat{y}$ and the error $\hat{y}-y$.
2. Compute $\frac{\partial \ell}{\partial w}=2(\hat{y}-y)x$ and $\frac{\partial \ell}{\partial b}=2(\hat{y}-y)$.
3. Update $w$ and $b$ using left-arrow notation.
4. After the update, check numerically whether the prediction for $x=2$ moves closer to 5.
5. Draw the loss as a bowl-shaped curve and show a gradient descent step moving downhill.
