# Bag of Questions — Session 2 — af

## Question: Logistic Regression from Scratch — Fit Function

You are implementing logistic regression from scratch using NumPy.

The class has these attributes:

```python
self.lr
self.n_iters
self.weights
self.bias
```

1. Write the `_sigmoid` function in Python.
2. Write the main steps inside the `fit(X, y)` function:
   - get `n_samples` and `n_features`,
   - initialize weights and bias,
   - compute the linear model,
   - apply sigmoid,
   - compute `dw` and `db`,
   - update weights and bias.
3. In your code, use `np.dot(X, self.weights) + self.bias` for the linear model.
4. In your code, use:

   ```python
   dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
   db = (1 / n_samples) * np.sum(y_predicted - y)
   ```

5. Explain why `dw` has the same shape as `self.weights`.
6. Explain why `db` is a scalar.

## Question: Predict Function

1. Write the `predict(X)` function for logistic regression from scratch.
2. Your function should:
   - compute the linear model,
   - apply sigmoid,
   - convert probabilities to class labels using threshold 0.5,
   - return a NumPy array of predicted classes.
3. If predicted probabilities are `[0.2, 0.5, 0.5001, 0.9]`, what class labels are returned by the course implementation using `1 if i > 0.5 else 0`?
