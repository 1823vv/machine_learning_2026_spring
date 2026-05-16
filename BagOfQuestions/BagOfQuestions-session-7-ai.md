## Question: Training Mode versus Evaluation Mode

Some neural-network layers behave differently during training and evaluation. Dropout and batch normalization are two important examples.

1. Explain what dropout does during training.
2. Explain what dropout should do during evaluation.
3. Explain what batch normalization uses during training.
4. Explain what batch normalization should use during inference.
5. Draw a two-column table comparing train mode and evaluation mode for dropout, batch normalization, and ordinary dense layers.
6. What can go wrong if dropout is accidentally left on during validation or testing?
7. What can go wrong if validation metrics are computed while the model is in the wrong mode?

## Question: Running Statistics in Batch Normalization

Batch normalization can maintain running estimates of mean and variance for inference:

$$
\mu_{running} \leftarrow (1-\alpha)\mu_{running}+\alpha\mu_B,
$$

$$
\sigma^2_{running} \leftarrow (1-\alpha)\sigma^2_{running}+\alpha\sigma_B^2.
$$

1. Explain the meaning of $\mu_B$ and $\sigma_B^2$.
2. Explain the meaning of the running estimates.
3. Why are running estimates useful during inference?
4. Why would using statistics from a single test batch be unstable?
5. What is the role of $\alpha$ in the running-statistics update?
