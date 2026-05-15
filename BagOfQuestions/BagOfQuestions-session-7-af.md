## Question: Hyperparameters Are Not Parameters

A student trains a neural network and adjusts learning rate, dropout probability, L2 strength, batch size, and number of hidden layers.

1. Define model parameters.
2. Define hyperparameters.
3. Classify each item as parameter or hyperparameter:
   - weights $W$,
   - biases $b$,
   - learning rate $\eta$,
   - dropout probability $p$,
   - L2 strength $\lambda$,
   - batch size,
   - number of layers.
4. Write the usual parameter-learning objective:

   $$
   \theta^* = \arg\min_\theta \mathcal{L}(\theta).
   $$

5. Write the hyperparameter-selection objective using validation loss:

   $$
   \lambda^* = ?
   $$

6. Explain why validation loss, not training loss, is used for hyperparameter selection.
7. Draw a two-level diagram:
   - inner loop: train parameters,
   - outer loop: choose hyperparameters.

## Question: Bad Hyperparameters, Different Failures

For each case below, explain what might happen to training and validation performance:

1. learning rate too large,
2. learning rate too small,
3. dropout probability too high,
4. L2 strength too small,
5. model depth too large for a small dataset.
