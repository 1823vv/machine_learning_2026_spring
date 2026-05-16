## Question: Binary Cross-Entropy Loss

For binary logistic regression, the model outputs a probability $\hat{y}^{(i)} \in (0,1)$ for each sample $i$, and the label is $y^{(i)} \in \{0,1\}$.

1. Write the binary cross-entropy loss for one training example and the average binary cross-entropy loss over $n$ examples.
2. Simplify the one-example BCE loss for $y=1$ and for $y=0$.
3. Explain why BCE gives a very large penalty when the model is confidently wrong.
4. Draw two curves on the same diagram: loss when $y=1$ as a function of $\hat{y}$, and loss when $y=0$ as a function of $\hat{y}$. Mark where each loss is small or large.

## Question: BCE versus MSE

For logistic regression, one could imagine using mean squared error on the predicted probabilities:

$$
\mathcal{L}_{\mathrm{MSE}}=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}^{(i)}-y^{(i)})^2.
$$

1. Write the MSE loss formula above from memory.
2. Explain two reasons why BCE is usually preferred over MSE for logistic regression.
3. In particular, explain what happens to the MSE gradient signal when sigmoid output is saturated near 0 or 1.
