## Question: L1 vs L2 Regularization

In linear regression, we can add regularization terms to the loss function to prevent overfitting. Consider a 2D weight vector $W = [w_1, w_2]^T$ and a regularization parameter $\lambda > 0$.

1. Write the L1 regularization penalty term and the L2 regularization penalty term for the weight vector $w$.

2. Draw two separate 2D coordinate systems with axes $w_1$ and $w_2$:
   - In the first plot, draw the region where the L1 penalty is less than or equal to a constant $t$ (i.e., $|w_1| + |w_2| \leq t$).
   - In the second plot, draw the region where the L2 penalty is less than or equal to a constant $t$ (i.e., $w_1^2 + w_2^2 \leq t^2$).

3. Based on your drawings, explain why L1 regularization tends to produce exactly zero weights (sparse solutions) while L2 regularization only shrinks weights toward zero without making them exactly zero.

4. In one sentence, explain how regularization helps prevent overfitting.