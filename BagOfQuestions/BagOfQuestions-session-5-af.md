## Question: Adam Formulas — First Moment, Second Moment, Update

Adam tracks two moving averages for a gradient $g$.

1. Write the first moment update formula.
2. Write the second moment update formula.
3. Explain why Adam's $m$ is similar to momentum.
4. Explain why Adam's $v$ is not the same as the momentum velocity $v$.
5. Write the bias-corrected first moment formula.
6. Write the bias-corrected second moment formula.
7. Write the Adam parameter update rule.
8. Explain the role of $\epsilon$ in the Adam update.
9. List typical default values for $\eta$, $\beta_1$, $\beta_2$, and $\epsilon$.

## Question: Adam Intuition

Draw an optimization landscape where different directions have different gradient scales.

1. How does the first moment help with noisy gradients?
2. How does the second moment adapt step sizes per parameter?
3. Why can Adam be less sensitive to the learning rate than plain SGD?
