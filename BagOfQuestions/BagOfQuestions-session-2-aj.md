# Bag of Questions — Session 2 — aj

## Question: Debugging a Logistic Regression Implementation

A student writes this logistic regression training code:

```python
linear_model = np.dot(X, self.weights) + self.bias
y_predicted = linear_model

dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
db = (1 / n_samples) * np.sum(y_predicted - y)

self.weights -= self.lr * dw
self.bias -= self.lr * db
```

1. What important logistic regression step is missing?
2. Write the corrected two lines that should compute the probability prediction.
3. Explain why the missing step matters for binary classification.
4. If `linear_model` contains values `[-10, 0, 10]`, what range problem can happen if we use it directly as a probability?

## Question: Compare Linear and Logistic Regression Code

1. In one short paragraph, explain what stays almost the same when moving from linear regression from scratch to logistic regression from scratch.
2. In one short paragraph, explain what must change.
3. Fill in the blanks in this logistic regression skeleton:

```python
linear_model = np.dot(X, self.weights) + self.bias
y_predicted = ________(linear_model)

dw = (1 / n_samples) * np.dot(X.T, (________ - y))
db = (1 / n_samples) * np.sum(________ - y)
```

4. Why does `predict` return class labels, while the sigmoid output itself gives probabilities?
5. Draw a small diagram comparing:

   ```text
   Linear Regression:   X -> linear model -> real-valued prediction
   Logistic Regression: X -> linear model -> sigmoid -> probability -> class label
   ```
