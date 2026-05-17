## Question: Logistic Regression Fit Method and Internal Sigmoid

We are implementing binary logistic regression `class MyOwnLogisticRegression` from scratch with NumPy. The implementation initializes one weight per feature, computes a linear score, applies the internal `_sigmoid` helper, and updates parameters by gradient descent.

Fill in the `____YOUR_CODE_HERE__N_____` blanks in this code skeleton.

```python
class MyOwnLogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return ____YOUR_CODE_HERE__1_____

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(____YOUR_CODE_HERE__2_____)
        self.bias = ____YOUR_CODE_HERE__3_____

        for _ in range(self.n_iters):
            linear_model = np.dot(____YOUR_CODE_HERE__4_____, ____YOUR_CODE_HERE__5_____) + self.bias
            y_predicted = self._sigmoid(____YOUR_CODE_HERE__6_____)

            dw = (1 / n_samples) * np.dot(____YOUR_CODE_HERE__7_____, (____YOUR_CODE_HERE__8_____ - y))
            db = (1 / n_samples) * np.sum(____YOUR_CODE_HERE__9_____ - y)

            self.weights -= self.lr * ____YOUR_CODE_HERE__10_____
            self.bias -= self.lr * ____YOUR_CODE_HERE__11_____
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
- `____YOUR_CODE_HERE__11_____`:

Then answer the following question:

12. Explain why the gradient expression uses `(y_predicted - y)` for binary cross-entropy with sigmoid.
