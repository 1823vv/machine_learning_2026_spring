## Question: From Gradient to Parameter Update

During neural-network training, suppose the loss over the current mini-batch is $\mathcal{L}$ and the gradient for a weight matrix is

$$
g = \frac{\partial \mathcal{L}}{\partial W}.
$$

The learning rate is $\eta$, and gradient descent uses the update notation

$$
W \leftarrow W - \eta g.
$$

1. Explain why computing a gradient is not the same thing as learning.
2. In one sentence, explain why the update moves in the negative gradient direction.
3. Draw a one-dimensional loss curve and label the current parameter value, the gradient direction, the negative-gradient direction, and the updated parameter value.
4. Explain the role of the learning rate $\eta$.
5. If one scalar weight has value $W=2.0$, gradient $g=0.5$, and learning rate $\eta=0.1$, compute the updated value of $W$.
6. If one scalar weight has value $W=2.0$, gradient $g=-0.5$, and learning rate $\eta=0.1$, compute the updated value of $W$.
7. Explain why the sign of the gradient changes the update direction.

## Question: Learning Rate Intuition

When training a machine-learning model by gradient descent, the learning rate $\eta$ controls the step size in each parameter update. The same model and dataset can behave very differently for different choices of $\eta$.

1. What can happen if the learning rate is too small?
2. What can happen if the learning rate is too large?
3. What does it mean for a learning rate to be reasonably chosen?
4. Draw three optimization paths on a simple bowl-shaped loss curve: too slow, stable, and unstable.
5. Why is the learning rate a hyperparameter rather than a learned parameter?
