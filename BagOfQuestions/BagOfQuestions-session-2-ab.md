## Question: The Sigmoid Function as a Probability Gate

A logistic regression model computes:

$$
z = xW + b
$$

and then applies:

$$
\hat{y} = \sigma(z)
$$

1. Write the formula for $\sigma(z)$.
2. State the range of $\sigma(z)$.
3. Compute $\sigma(0)$ exactly.
4. Explain, in words, what it means if $\hat{y}=0.93$ for a binary classifier.
5. Explain, in words, what it means if $\hat{y}=0.51$.
6. Draw the rough shape of the sigmoid curve. Your drawing should show:
   - the horizontal axis $z$,
   - the vertical axis $\sigma(z)$,
   - the point $(0,0.5)$,
   - the saturation regions near 0 and 1.
7. Explain why sigmoid saturation can make learning slower for very confident predictions.

## Question: Sigmoid Derivative

1. Write the formula for the derivative of the sigmoid function.
2. Rewrite the derivative using $\hat{y}$ instead of $\sigma(z)$.
3. For which value of $\hat{y}$ is $\hat{y}(1-\hat{y})$ largest: near 0, near 0.5, or near 1?
4. Explain why this fact is relevant for gradient-based learning.
