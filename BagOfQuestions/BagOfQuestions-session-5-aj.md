## Question: Debugging Neural-Network Optimization Without Writing New Optimizer Code

A neural network is not learning: the training loss is almost constant for many epochs. You are not asked to write a new optimizer. Instead, reason about possible optimization and implementation problems.

1. Give two possible learning-rate problems.
2. Give one possible data preprocessing problem.
3. Give one possible initialization problem.
4. Give one possible activation-function problem.
5. Give one possible label/loss mismatch problem.
6. Explain how checking whether gradients are all zero or extremely large could help.
7. Draw a debugging checklist from input data to predictions to loss to gradients to parameter updates.

## Question: Assertions and Breakpoints for Shape Debugging

In a neural-network implementation using row-vector batches, a mini-batch activation matrix often has shape $A^{(l-1)} \in \mathbb{R}^{B \times n_{l-1}}$, and a dense layer weight matrix has shape $W^{(l)} \in \mathbb{R}^{n_{l-1} \times n_l}$.

1. What should the shape of $Z^{(l)}=A^{(l-1)}W^{(l)}+\mathbf{1}b^{(l)}$ be?
2. Why are shape checks useful before blaming the optimizer?
3. Give two examples of assertions you might conceptually check, without writing code.
4. Explain why NaN or infinite losses are often optimization red flags.
5. Where in the training loop would you inspect predictions, loss, gradients, and updated parameters?
