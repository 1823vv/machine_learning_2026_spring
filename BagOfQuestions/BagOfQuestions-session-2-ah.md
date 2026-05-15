# Bag of Questions — Session 2 — ah

## Question: Build the Logistic Regression Pipeline Diagram

Draw a complete training-time diagram for logistic regression on a batch of data.

Your diagram should include these objects:

- $X$,
- $W$,
- $b$,
- $Z=XW+b$,
- $\hat{Y}=\sigma(Z)$,
- $Y$,
- binary cross-entropy loss,
- gradients $\partial \mathcal{L}/\partial W$ and $\partial \mathcal{L}/\partial b$,
- updates of $W$ and $b$.

Then answer:

1. Which arrow corresponds to the linear part of the model?
2. Which arrow corresponds to the nonlinear probability transformation?
3. Which step converts model output into a training objective?
4. Which step changes the parameters?
5. Why do we call $\hat{Y}$ probabilities instead of final class labels during training?

## Question: Formula Memory Check

Write the following formulas without explanation:

1. $z$ for one sample in row-vector convention.
2. $\hat{y}$ for one sample.
3. Binary cross-entropy loss for $n$ samples.
4. Gradient of BCE with respect to $W$ in batch matrix form.
5. Gradient of BCE with respect to $b$ in batch matrix form or summation form.
6. Gradient descent update for $W$.
7. Gradient descent update for $b$.
