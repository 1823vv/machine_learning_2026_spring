## Question: Compare GD, Mini-Batch SGD, Momentum, and Adam

Several optimizers can be viewed as different ways to use gradient information to update parameters. Assume $g$ denotes a gradient or mini-batch gradient for $W$.

1. Write the basic gradient descent update rule.
2. Explain how mini-batch SGD changes how $g$ is computed.
3. Write the momentum update formulas.
4. Write the main Adam moment formulas and final update rule.
5. Draw four rough optimization paths: full-batch GD, mini-batch SGD, momentum, and Adam.
6. Which method is usually smoothest and most expensive per update when the dataset is large?
7. Which methods are designed to handle noisy gradients better?
8. Why is there no single optimizer that is always best for every problem?

## Question: Choosing an Optimizer

A practitioner must choose an optimizer for training a neural network, but optimizer choice depends on the dataset, model, batch size, learning rate, and generalization behavior.

1. Give one reason Adam is often a good first optimizer to try.
2. Give one reason SGD with momentum may still be preferred in some settings.
3. Why should optimizer choice be evaluated using validation performance, not only training loss?
4. What other hyperparameters interact with optimizer choice?
5. Draw a small workflow for comparing two optimizers fairly.
