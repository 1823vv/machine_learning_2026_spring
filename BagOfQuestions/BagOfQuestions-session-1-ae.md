## Question: Linear Regression from Scratch

We are implementing linear regression `class MyOwnLinearRegression` from scratch with NumPy. The model receives a feature matrix `X` with shape `(n_samples, n_features)` and a target vector `y` with shape `(n_samples,)`. In mathematical notation, this corresponds to $X \in \mathbb{R}^{n \times d}$ and $Y \in \mathbb{R}^{n \times 1}$. The batch prediction rule is

$$
\hat{Y}=XW+\mathbf{1}b.
$$

Fill in the `____YOUR_CODE_HERE__N_____` blanks in the code skeleton below.

```python
class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=30000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(____YOUR_CODE_HERE__1_____)
        self.bias = ____YOUR_CODE_HERE__2_____

        for _ in range(self.n_iters):
            y_predicted = np.dot(____YOUR_CODE_HERE__3_____, ____YOUR_CODE_HERE__4_____) + self.bias

            dw = (2 / n_samples) * np.dot(____YOUR_CODE_HERE__5_____, (y_predicted - y))
            db = (2 / n_samples) * np.sum(____YOUR_CODE_HERE__6_____)

            self.weights = self.weights - self.lr * ____YOUR_CODE_HERE__7_____
            self.bias = self.bias - self.lr * ____YOUR_CODE_HERE__8_____

    def predict(self, X):
        y_predicted = np.dot(____YOUR_CODE_HERE__9_____, ____YOUR_CODE_HERE__10_____) + self.bias
        return y_predicted
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
- `____YOUR_CODE_HERE__10_____`:

Then answer the following short questions:

1. Why should `self.weights` have one entry for each feature, and what is the role of `self.bias`?
2. Why do we use `np.dot(X, self.weights)` instead of elementwise multiplication `X * self.weights` for prediction?
3. In the line for `dw`, why do we use `X.T`, and what shape should `dw` have?
