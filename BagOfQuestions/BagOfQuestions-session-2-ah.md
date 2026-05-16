## Question: Logistic Regression Formula Checklist

For binary logistic regression, use the row-vector convention with one example $x \in \mathbb{R}^{1 \times d}$, weights $W \in \mathbb{R}^{d \times 1}$, bias $b \in \mathbb{R}^{1 \times 1}$, and batch matrices $X \in \mathbb{R}^{n \times d}$, $Y \in \mathbb{R}^{n \times 1}$, and $\hat{Y} \in \mathbb{R}^{n \times 1}$.

Write the following formulas without explanation:

1. The logit $z$ and probability $\hat{y}$ for one sample.
2. The binary cross-entropy loss for $n$ samples.
3. The gradient of BCE with respect to $W$ in batch matrix form.
4. The gradient of BCE with respect to $b$ in batch matrix form or summation form.
5. The gradient descent updates for $W$ and $b$ using left-arrow notation.
