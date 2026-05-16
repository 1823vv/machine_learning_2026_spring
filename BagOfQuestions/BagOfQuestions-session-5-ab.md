## Question: Learning Rate — Too Small, Too Large, Just Right

A model is trained by gradient descent with update

$$
W \leftarrow W - \eta g,
$$

where $g = \partial \mathcal{L}/\partial W$ and $\eta$ is the learning rate.

1. Explain what happens when $\eta$ is too small.
2. Explain what happens when $\eta$ is too large. Explain what happens when $\eta$ is reasonably chosen.
3. Draw three optimization paths on a bowl-shaped loss surface: slow progress, overshooting/divergence, and stable convergence. Why can the same learning rate work for one model but fail for another?
4. Give typical learning-rate values that might be tried for simple gradient descent and for Adam.

## Question: Learning Rate Schedule

Instead of using a constant learning rate for all epochs, many training procedures use a learning-rate schedule, where $\eta$ changes during training.

1. Why might a relatively large learning rate be useful early in training?
2. Why might a smaller learning rate be useful later in training? Draw a learning-rate schedule that decreases over epochs.
3. Explain how decreasing $\eta$ can help the optimizer settle near a minimum.
4. What could go wrong if the learning rate is decreased too aggressively too early?
