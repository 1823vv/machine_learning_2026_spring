## Question: Compare GD, Mini-Batch SGD, Momentum, and Adam

Several optimizers can be viewed as different ways to use gradient information to update parameters. Assume $g$ denotes a gradient or mini-batch gradient for $W$.

1. Write the basic gradient descent update rule. Explain how mini-batch SGD changes how $g$ is computed.
2. Write the momentum update formulas. Write the main Adam moment formulas and final update rule.
3. Draw four rough optimization paths: full-batch GD, mini-batch SGD, momentum, and Adam. Which method is usually smoothest and most expensive per update when the dataset is large?
4. Which methods are designed to handle noisy gradients better? Why is there no single optimizer that is always best for every problem?

## Question: Choosing an Optimizer

A practitioner must choose an optimizer for training a neural network, but optimizer choice depends on the dataset, model, batch size, learning rate, and generalization behavior.

1. Give one reason Adam is often a good first optimizer to try.
2. Give one reason SGD with momentum may still be preferred in some settings. Why should optimizer choice be evaluated using validation performance, not only training loss?
3. What other hyperparameters interact with optimizer choice?
4. Draw a small workflow for comparing two optimizers fairly.
