## Question: Linear Regression from Scratch 

We are implementing linear regression `class MyOwnLinearRegression` from scratch with NumPy. TODO BY AI: Make this look like the ## Question Fill the the `____YOUR_CODE_HERE__N_____` in BagOfQuestions-session-4.md .

```python
class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=30000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)
            self.weights  = self.weights - self.lr * dw
            self.bias  = self.bias -  self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
```

