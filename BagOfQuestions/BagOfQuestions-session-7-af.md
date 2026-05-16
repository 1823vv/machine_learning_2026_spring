## Question: Hyperparameters Are Not Parameters

A model parameter is learned directly from training data, while a hyperparameter is chosen outside the training update and selected using validation performance.

1. Give three examples of learned parameters.
2. Give five examples of hyperparameters.
3. Explain why learning rate $\eta$ is a hyperparameter.
4. Explain why dropout probability $p$ is a hyperparameter.
5. Write the usual parameter-learning objective:

   $$
   W^* = \arg\min_W \mathcal{L}_{train}(W;\lambda).
   $$

6. Write a hyperparameter-selection objective using validation loss:

   $$
   \lambda^* = \arg\min_\lambda \mathcal{L}_{val}(W^*(\lambda);\lambda).
   $$

7. Draw a two-level diagram: inner loop trains parameters, outer loop chooses hyperparameters.
8. Explain why validation loss, not training loss, is used for hyperparameter selection.

## Question: Bad Hyperparameters, Different Failures

Different bad hyperparameter choices can cause different training or generalization failures.

For each case below, explain what might happen to training and validation performance:

1. learning rate too large,
2. learning rate too small,
3. dropout probability too high,
4. L2 strength too small,
5. model depth too large for a small dataset,
6. batch size extremely small.
