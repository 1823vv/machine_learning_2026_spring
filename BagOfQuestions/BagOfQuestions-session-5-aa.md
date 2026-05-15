## Question: From Gradient to Parameter Update

After backpropagation, a neural network has computed a gradient:

$$
g = \frac{\partial \mathcal{L}}{\partial W}
$$

1. Explain why computing a gradient is not the same as learning.
2. Write the basic gradient descent update rule using learning rate $\eta$.
3. In one sentence, explain why we move in the negative gradient direction.
4. Draw a one-dimensional loss curve and show:
   - current parameter value,
   - gradient direction,
   - negative gradient direction,
   - updated parameter value.
5. Explain the role of the learning rate in the update.
6. If $W=2.0$, $g=0.5$, and $\eta=0.1$, compute the updated $W$.
7. If $W=2.0$, $g=-0.5$, and $\eta=0.1$, compute the updated $W$.
8. Explain why the sign of the gradient matters.


## Question: Learning Rate Intuition

1. What can happen if the learning rate is too small?
2. What can happen if the learning rate is too large?
3. Draw three optimization paths: too slow, good, and unstable.