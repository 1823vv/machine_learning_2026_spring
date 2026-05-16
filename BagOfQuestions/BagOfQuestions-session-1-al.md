## Question: Linear Regression Predict Method and Shape Debugging

We are implementing linear regression `class MyOwnLinearRegression` from scratch with NumPy. After training, `self.weights.shape == (n_features,)` and `self.bias` is a scalar. The `predict` method should return one prediction per input row.

Fill in the `____YOUR_CODE_HERE__N_____` blanks in the method below.

```python
def predict(self, X):
    y_predicted = np.dot(____YOUR_CODE_HERE__1_____, ____YOUR_CODE_HERE__2_____) + ____YOUR_CODE_HERE__3_____
    return ____YOUR_CODE_HERE__4_____
```

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
- `____YOUR_CODE_HERE__3_____`:
- `____YOUR_CODE_HERE__4_____`:

Then answer the following short questions:

1. If `X.shape == (20, 3)` and `self.weights.shape == (3,)`, what shape should `y_predicted` have?
2. Why is `np.dot(X, self.weights)` the correct operation for multiple linear regression, while `X * self.weights` is not the desired prediction rule?
3. If a plotting function creates `X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)`, explain why the resulting prediction has 100 values.
