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
2. In one sentence, explain why the update moves in the negative gradient direction. Draw a one-dimensional loss curve and label the current parameter value, the gradient direction, the negative-gradient direction, and the updated parameter value.
3. Explain the role of the learning rate $\eta$. If one scalar weight has value $W=2.0$, gradient $g=0.5$, and learning rate $\eta=0.1$, compute the updated value of $W$.
4. If one scalar weight has value $W=2.0$, gradient $g=-0.5$, and learning rate $\eta=0.1$, compute the updated value of $W$. Explain why the sign of the gradient changes the update direction.
