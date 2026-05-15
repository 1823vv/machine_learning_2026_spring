## Question: Linear Regression from Scratch — `fit` Function

You are implementing linear regression from scratch with NumPy.

The class has:

```python
self.lr
self.n_iters
self.weights
self.bias
```

1. Write the initialization of `self.weights` and `self.bias` inside `fit(X, y)`.
2. Write the line that computes predictions using `np.dot`.
3. Write the formula/code for `dw`.
4. Write the formula/code for `db`.
5. Write the parameter update lines.
6. Explain why `self.weights` should have shape `(n_features,)` in this implementation.
7. Explain why `db` is a scalar.
8. Draw the flow of the `fit` function:

   ```text
   initialize -> predict -> compute gradients -> update -> repeat
   ```

## Question: `predict` Function

1. Write the `predict(X)` function for this class.
2. Should `predict(X)` update the weights? Explain.
3. If `X.shape == (20, 3)` and `self.weights.shape == (3,)`, what is the shape of the prediction output?
