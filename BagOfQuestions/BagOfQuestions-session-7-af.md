## Question: Model Parameters versus Hyperparameters

In machine learning frameworks, parameters and hyperparameters serve fundamentally different roles during optimization and model selection.

1. Contrast the core mechanism by which a model **parameter** is determined with how a **hyperparameter** is chosen.
2. Provide two specific examples of learned parameters in a neural network architecture.
3. Provide two specific examples of model or optimization hyperparameters.
4. Explain why the learning rate $\eta$ cannot be updated directly using the same training gradient descent steps applied to the network weights.
5. Explain why the dropout probability $p$ cannot be optimized by minimizing the training loss function.
6. Explain why hyperparameter selection must be guided by validation dataset performance rather than training dataset performance.


## Question: Hyperparameter Configurations and Failure Modes

Improperly configured hyperparameters can cause distinct training or generalization failures, leading to problems such as divergence, severe overfitting, or underfitting.

Describe the typical impact on both training performance and validation performance for each of the following scenarios:

1. The learning rate $\eta$ is set to an excessively large value.
2. The learning rate $\eta$ is set to an excessively small, non-zero value.
3. The dropout drop probability $p$ is set too close to 1.
4. The L2 regularization penalty strength $\lambda$ is set too close to 0.
5. The neural network depth and capacity are exceptionally large relative to a very small training dataset.
6. The mini-batch size used for stochastic optimization is set to an extremely small value (e.g., a batch size of 1).
