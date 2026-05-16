## Question: Logistic Regression from Scratch

We are implementing binary logistic regression `class MyOwnLogisticRegression` from scratch with NumPy. The model receives a feature matrix `X` with shape `(n_samples, n_features)` and a binary target vector `y` with values in `{0, 1}`. First we compute the linear score

$$
z=XW+\mathbf{1}b,
$$

then we convert it to a probability with the sigmoid function

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Fill in the `____YOUR_CODE_HERE__N_____` blanks in the code skeleton below.

```python
class MyOwnLogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=10000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return ____YOUR_CODE_HERE__1_____

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(____YOUR_CODE_HERE__2_____)
        self.bias = ____YOUR_CODE_HERE__3_____

        for _ in range(self.n_iters):
            linear_model = np.dot(____YOUR_CODE_HERE__4_____, ____YOUR_CODE_HERE__5_____) + self.bias
            y_predicted = self.sigmoid(____YOUR_CODE_HERE__6_____)

            dw = (1 / n_samples) * np.dot(____YOUR_CODE_HERE__7_____, (y_predicted - y))
            db = (1 / n_samples) * np.sum(____YOUR_CODE_HERE__8_____)

            self.weights = self.weights - self.lr * ____YOUR_CODE_HERE__9_____
            self.bias = self.bias - self.lr * ____YOUR_CODE_HERE__10_____

    def predict(self, X):
        linear_model = np.dot(____YOUR_CODE_HERE__11_____, ____YOUR_CODE_HERE__12_____) + self.bias
        y_probability = self.sigmoid(____YOUR_CODE_HERE__13_____)
        y_predicted_class = [1 if p >= 0.5 else 0 for p in ____YOUR_CODE_HERE__14_____]
        return np.array(y_predicted_class)
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
- `____YOUR_CODE_HERE__12_____`:
- `____YOUR_CODE_HERE__13_____`:
- `____YOUR_CODE_HERE__14_____`:

Then answer the following short questions:

1. Why does logistic regression need the sigmoid function after the linear score, and what does `y_probability` mean before it is converted into class labels?
2. Why is the gradient coefficient `(1 / n_samples)` here instead of `(2 / n_samples)` as in the usual MSE formula for linear regression?
3. What probability threshold is used in this `predict` function?
