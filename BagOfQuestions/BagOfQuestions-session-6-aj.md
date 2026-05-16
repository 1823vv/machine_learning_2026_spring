## Question: Bivariate Normal Distribution and Covariance Matrix

A bivariate normal distribution describes two continuous random variables, such as $X_1$ and $X_2$, jointly. Its covariance matrix can be written as

$$
\Sigma = \begin{bmatrix}
\sigma_1^2 & \sigma_{12} \\
\sigma_{12} & \sigma_2^2
\end{bmatrix}.
$$

1. What do the diagonal entries of $\Sigma$ represent?
2. What do the off-diagonal entries represent? If $\sigma_{12}>0$, what visual pattern would you expect in a scatter plot?
3. If $\sigma_{12}<0$, what visual pattern would you expect? If $\sigma_{12}=0$, what visual pattern would you expect?
4. Draw scatter plots for positive, negative, and near-zero covariance. Explain the difference between covariance and correlation.

## Question: Why Probability Distributions Matter for Model Selection

Evaluation metrics are computed on finite datasets, so validation performance can vary depending on which examples are sampled. Thinking probabilistically helps us reason about uncertainty.

1. Why might validation accuracy vary if we resample the validation set?
2. Why is a larger validation set usually more stable than a tiny validation set? How can cross-validation reduce dependence on one lucky or unlucky split?
3. Why should small differences in validation score be interpreted carefully?
4. Draw a simple picture of two models whose validation-score distributions overlap.
