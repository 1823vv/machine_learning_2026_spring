## Question: L1 and L2 Regularization Formulas

Regularization adds a penalty to the loss to discourage overly complex models. Let $W_j$ denote a weight parameter and $\lambda$ denote regularization strength.

1. Write a regularized objective using an empirical loss $\mathcal{L}_{data}(W)$ plus an L1 penalty.
2. Write a regularized objective using $\mathcal{L}_{data}(W)$ plus an L2 penalty. Explain the role of $\lambda$.
3. What happens when $\lambda=0$? What can happen when $\lambda$ is extremely large?
4. Explain why regularization can improve validation performance even if it increases training loss.

## Question: Large Weights

A model with very large weights can be sensitive to small input changes. Regularization can discourage such large weights.

1. Explain why large weights may indicate a more complex or more sensitive model.
2. Which penalty, L1 or L2, is especially associated with sparse weights? Which penalty, L1 or L2, smoothly shrinks many weights toward zero?
3. Draw a simple comparison of unregularized weights and regularized weights.
4. Explain why regularization strength should be selected using validation data.
