# Bag of Questions — Session 2 — ad

## Question: Binary Cross-Entropy Loss

For binary logistic regression, the model outputs a probability $\hat{y}^{(i)} \in (0,1)$ for each sample $i$, and the label is $y^{(i)} \in \{0,1\}$.

1. Write the binary cross-entropy loss for one training example.
2. Write the average binary cross-entropy loss over $n$ training examples.
3. If $y=1$, simplify the one-example BCE loss.
4. If $y=0$, simplify the one-example BCE loss.
5. Explain why BCE gives a very large penalty when the model is confidently wrong.
6. Draw two curves on the same diagram (an idea diagram):
   - loss when $y=1$ as a function of $\hat{y}$,
   - loss when $y=0$ as a function of $\hat{y}$.
7. On your drawing, mark where the loss is small and where the loss is large.

## Question: BCE versus MSE

For logistic regression, one could imagine using MSE:

$$
\mathcal{L}_{\mathrm{MSE}}=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}^{(i)}-y^{(i)})^2
$$

1. Write the MSE loss formula above from memory.
2. Explain two reasons why BCE is usually preferred over MSE for logistic regression.
3. In particular, explain what happens to the MSE gradient signal when sigmoid output is saturated near 0 or 1.
