## Question: Linear Regression Fit Method Details

We are implementing linear regression `class MyOwnLinearRegression` from scratch with NumPy. The implementation shape is `X.shape == (n_samples, n_features)` and `y.shape == (n_samples,)`. The mathematical batch convention is $X \in \mathbb{R}^{n \times d}$ and $Y \in \mathbb{R}^{n \times 1}$.

Fill in the `____YOUR_CODE_HERE__N_____` blanks in this `fit` fragment.

```python
def fit(self, X, y):
    n_samples, n_features = X.shape

    self.weights = np.zeros(____YOUR_CODE_HERE__1_____)
    self.bias = ____YOUR_CODE_HERE__2_____

    for _ in range(self.n_iters):
        y_predicted = np.dot(____YOUR_CODE_HERE__3_____, ____YOUR_CODE_HERE__4_____) + self.bias

        dw = (2 / n_samples) * np.dot(____YOUR_CODE_HERE__5_____, (____YOUR_CODE_HERE__6_____ - y))
        db = (2 / n_samples) * np.sum(____YOUR_CODE_HERE__7_____ - y)

        self.weights = self.weights - self.lr * ____YOUR_CODE_HERE__8_____
        self.bias = self.bias - self.lr * ____YOUR_CODE_HERE__9_____
```

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
- `____YOUR_CODE_HERE__3_____`:
- `____YOUR_CODE_HERE__4_____`:
- `____YOUR_CODE_HERE__5_____`:
- `____YOUR_CODE_HERE__6_____`:
- `____YOUR_CODE_HERE__7_____`:
- `____YOUR_CODE_HERE__8_____`:
- `____YOUR_CODE_HERE__9_____`:

Then answer the following short questions:

1. What are the shapes of `self.weights`, `y_predicted`, `dw`, and `db`?
2. Why does the gradient use the factor `(2 / n_samples)` for this mean-squared-error convention?
3. Explain why both updates subtract `self.lr` times the corresponding gradient.
