## Question: Adam Formulas — First Moment, Second Moment, Update

Adam adapts gradient descent by tracking two moving averages for a gradient $g$: a first moment $m$ and a second moment $v$. Use the update notation with learning rate $\eta$.

1. Write the first moment update formula. Write the second moment update formula.
2. Explain why Adam's $m$ is similar to momentum. Explain why Adam's $v$ is not the same as the momentum velocity $v$.
3. Write the bias-corrected first moment formula $\hat m$. Write the bias-corrected second moment formula $\hat v$.
4. Write the Adam parameter update rule for $W$. Explain the role of $\epsilon$ in the Adam update. List typical default values for $\eta$, $\beta_1$, $\beta_2$, and $\epsilon$.

## Question: Adam Intuition

Consider an optimization problem where different parameter directions have very different gradient scales. Adam uses the first moment to smooth noisy gradients and the second moment to adjust step sizes by gradient magnitude.

1. How does the first moment help with noisy gradients?
2. How does the second moment help when different directions have different gradient scales? Draw an optimization landscape with one steep direction and one shallow direction.
3. Explain why Adam can be easier to tune than plain SGD in many practical neural-network problems.
4. Give one reason Adam is not always guaranteed to give the best final generalization.
