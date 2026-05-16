## Question: Shape Checking with One-Hot Labels and Softmax

A multiclass neural network uses row-vector batches. For a mini-batch of $n$ examples and $K$ classes, the logits matrix is

$$
Z \in \mathbb{R}^{n \times K}.
$$

After softmax, the predicted probability matrix is

$$
\hat{Y} \in \mathbb{R}^{n \times K}.
$$

The one-hot target matrix is

$$
Y \in \mathbb{R}^{n \times K}.
$$

1. What does each row of $Y$ represent?
2. What should the sum of each row of $Y$ be for single-label multiclass classification? What should the sum of each row of $\hat{Y}$ be after softmax?
3. If $n=128$ and $K=10$, what are the shapes of $Z$, $\hat{Y}$, and $Y$? The softmax cross-entropy gradient often has the form $\frac{1}{n}(\hat{Y}-Y)$. What is its shape?
4. Why would a target matrix with shape $128 \times 1$ not match the one-hot version expected by this formula?
