## Question: Bivariate Normal Distribution and Covariance Matrix

Consider this code idea from Session 6:

```python
mu = [0, 0]
sigma = [[1, 0.5],
         [0.5, 1]]
```

This defines a bivariate normal distribution.

1. What does the vector `mu` represent?
2. What does the matrix `sigma` represent?
3. Explain the meaning of the diagonal entries of `sigma`.
4. Explain the meaning of the off-diagonal entries of `sigma`.
5. If the covariance is positive, what shape/orientation do you expect in the scatter plot?
6. If the covariance is negative, what shape/orientation do you expect in the scatter plot?
7. If the covariance is zero, what relationship do you expect between the two variables?
8. Draw three scatter plots for covariance $0.5$, $-0.5$, and $0$.

## Question: Why Probability Distributions Matter for Model Selection

1. Why is randomness important when we split data into train/validation/test sets?
2. Why can performance estimates vary depending on the random split?
3. How does cross-validation help reduce dependence on one random split?
4. Explain how understanding distributions can help interpret noisy evaluation results.
