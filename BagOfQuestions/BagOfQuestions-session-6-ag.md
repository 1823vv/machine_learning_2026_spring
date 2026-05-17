## Question: L1 vs L2 Regularization

In linear regression, we can add regularization terms to the loss function to prevent overfitting. Consider a 2D weight vector $W = [w_1, w_2]^T$ and a regularization parameter $\lambda > 0$. For simplicity, we ignore the scalar bias term $b$.

1. Write the L1 regularization penalty term and the L2 regularization penalty term for the weight vector $w$.

2. Draw two separate 2D coordinate systems with axes $w_1$ and $w_2$:
   - In the first plot, draw the region where the L1 penalty is less than or equal to a constant $t$ (i.e., $|w_1| + |w_2| \leq t$).
   - In the second plot, draw the region where the L2 penalty is less than or equal to a constant $t$ (i.e., $w_1^2 + w_2^2 \leq t^2$).
   - Overlay a set of elliptical loss contours. Mark the first point of contact where the expanding loss ellipse touches the regularized constraint region, 

3. Based on your drawings,  explain why L1 regularization tends to produce exactly zero weights (sparse solutions) while L2 regularization tends to only shrink weights toward zero without making them exactly zero.

4. While easily visualized in linear models, $L_1$ and $L_2$ regularization are heavily utilized in deep neural networks (where $L_2$ is often implemented as weight decay). Analyze the optimization behavior across these architectures at the boundaries of the hyperparameter space:
   * In a general sense, what happens to the network training and weight updates when $\lambda = 0$?
   * In a general sense, what happens to the learned weight parameters when $\lambda$ is set to an extremely large value ($\lambda \to \infty$)?

5. Explain why in deep learning L1 and L2 regularizations can improve validation performance even if it increases training loss.








