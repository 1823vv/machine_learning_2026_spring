## Question: Comparison of Gradient Optimization Algorithms

Optimization algorithms use gradient information to update parameters and minimize loss. Let $W$ be the weight parameter matrix, $\eta > 0$ be the learning rate, and $g$ be the gradient or mini-batch gradient calculated with respect to $W$.

1. Write out the standard mathematical update rule for basic gradient descent using the left-arrow ($\leftarrow$) notation. Explain how mini-batch Stochastic Gradient Descent (SGD) changes how the gradient vector $g$ is computed compared to full-batch Gradient Descent.
2. Write out the complete mathematical update formulas for **Momentum** optimization using a velocity variable $v$ and friction hyperparameter $\beta$. Next, write the formulas for **Adam** optimization, explicitly showing the moving averages of the first moment ($m$), the second moment ($v$), their bias corrections, and the final parameter update rule.
3. Sketch four separate optimization trajectories (paths) in a two-dimensional parameter space $(w_1, w_2)$ representing the convergence behavior of full-batch GD, mini-batch SGD, Momentum, and Adam. Assuming a massive dataset, state which of these methods produces the smoothest path and which requires the highest computational cost *per single parameter update*.
4. Identify which of these four optimization techniques are specifically engineered to mitigate the negative effects of high-variance, noisy gradient steps. Briefly explain the mechanism they use to achieve this stability.
