## Question: Adam Formulas — First Moment, Second Moment, Update

Adam adapts gradient descent by tracking two moving averages for a gradient $g$: a first moment $m$ and a second moment $v$. Use the update notation with learning rate $\eta$.

1. Write the first moment update formula. Write the second moment update formula.
2. Explain why Adam's $m$ is similar to momentum. Explain why Adam's $v$ is not the same as the momentum velocity $v$.
3. Write the bias-corrected first moment formula $\hat m$. Write the bias-corrected second moment formula $\hat v$.
4. Write the Adam parameter update rule for $W$. Explain the role of $\epsilon$ in the Adam update. List typical default values for $\eta$, $\beta_1$, $\beta_2$, and $\epsilon$.
