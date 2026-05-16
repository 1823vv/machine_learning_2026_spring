## Question: L1 and L2 Regularization — Formula, Geometry, and Intuition

Let $\mathcal{L}_{data}(W)$ be the data loss for a model and $\lambda$ be regularization strength.

1. Write the L1-regularized objective.
2. Write the L2-regularized objective.
3. Draw the L1 constraint shape for two weights.
4. Draw the L2 constraint shape for two weights.
5. Explain why L1 can produce sparse weights.
6. Explain why L2 tends to shrink weights smoothly.
7. Explain why $\lambda$ is a hyperparameter.
8. Why should $\lambda$ be chosen using validation performance?

## Question: Weight Penalty and Neural-Network Capacity

A neural network with very large weights can fit complex patterns and may overfit noisy data. Weight penalties constrain the effective capacity of the model.

1. Explain how a weight penalty can reduce overfitting.
2. Why might training loss increase after adding regularization?
3. Why can validation loss decrease even if training loss increases?
4. Draw training and validation loss with and without regularization.
5. Give one reason not to make regularization strength extremely large.
