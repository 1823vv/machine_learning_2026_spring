## Question: Compare GD, Mini-Batch SGD, Momentum, and Adam

Several optimizers can be viewed as different ways to use gradient information to update parameters. Assume $g$ denotes a gradient or mini-batch gradient for $W$.

1. Write the basic gradient descent update rule. Explain how mini-batch SGD changes how $g$ is computed.
2. Write the momentum update formulas. Write the main Adam moment formulas and final update rule.
3. Draw four rough optimization paths: full-batch GD, mini-batch SGD, momentum, and Adam. Which method is usually smoothest and most expensive per update when the dataset is large?
4. Which methods are designed to handle noisy gradients better? 