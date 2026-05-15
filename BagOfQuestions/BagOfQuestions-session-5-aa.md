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

## Question: Session 4 to Session 5 Story

In Session 4, `Dense.backward()` computed gradients and updated weights immediately. In Session 5, optimizers handle updates.

1. Explain why separating “gradient computation” from “parameter update” is a better design.
2. Draw a before/after diagram comparing the Session 4 design with the Session 5 design.
3. Which part should own gradients? Which part should own update rules?
