## Question: Learning Rate — Too Small, Too Large, Just Right

The learning rate $\eta$ controls the step size in:

$$
W \leftarrow W - \eta g
$$

1. Explain what happens when $\eta$ is too small.
2. Explain what happens when $\eta$ is too large.
3. Explain what happens when $\eta$ is reasonably chosen.
4. Draw three optimization paths on a bowl-shaped loss surface:
   - too small: slow progress,
   - too large: overshooting or divergence,
   - just right: stable convergence.
5. Why can the same learning rate work for one model but fail for another?
6. Why is learning rate called a hyperparameter rather than a parameter?
7. Give typical learning-rate values for simple gradient descent and Adam.

## Question: Learning Rate Schedule

1. Why might a large learning rate be useful early in training?
2. Why might a smaller learning rate be useful later in training?
3. Draw a learning-rate schedule that decreases over epochs.
4. Explain how decreasing $\eta$ can help the optimizer settle near a minimum.
