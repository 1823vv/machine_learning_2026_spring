## Question: The Sigmoid Function as a Probability Gate

A logistic regression model computes the logit $z=xW+b$ and then applies the sigmoid function $\hat{y}=\sigma(z)$.

1. Write the formula for $\sigma(z)$, state its range, and compute $\sigma(0)$ exactly.
2. Explain what $\hat{y}=0.93$ means for a binary classifier, and compare it with $\hat{y}=0.51$.
3. Draw the rough shape of the sigmoid curve, including the axes, the point $(0,0.5)$, and the saturation regions near 0 and 1.
4. Explain why sigmoid saturation can make learning slower for very confident predictions.

## Question: Sigmoid Derivative

The derivative of sigmoid is important for gradient-based learning in logistic regression and neural networks.

1. Write the derivative of the sigmoid function and rewrite it using $\hat{y}$ instead of $\sigma(z)$.
2. For which value of $\hat{y}$ is $\hat{y}(1-\hat{y})$ largest: near 0, near 0.5, or near 1?
3. Explain why this fact is relevant for gradient-based learning.
