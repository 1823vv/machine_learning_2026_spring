## Question: Hyperparameter Configurations and Failure Modes

Improperly configured hyperparameters can cause distinct training or generalization failures, leading to problems such as divergence, severe overfitting, or underfitting.

Describe the typical impact on both training performance and validation performance for each of the following scenarios:

1. The learning rate $\eta$ is set to an excessively large value.
2. The learning rate $\eta$ is set to an excessively small, non-zero value.
3. The dropout drop probability $p$ is set too close to 1.
4. The L2 regularization penalty strength $\lambda$ is set too close to 0.
5. The neural network depth and capacity are exceptionally large relative to a very small training dataset.
6. The mini-batch size used for stochastic optimization is set to an extremely small value (e.g., a batch size of 1).
