## Question: Compare Linear and Logistic Regression Code From Sratch

1. Explain what stays almost the same when moving from linear regression from scratch to logistic regression from scratch.
2. In one short paragraph, explain what must change.
3. Fill in the blanks in this logistic regression skeleton  (TODO: adopt the `____YOUR_CODE_HERE__N_____` as look in BagOfQuestions-session-4.md):

```python
linear_model = np.dot(X, self.weights) + self.bias
y_predicted = ________(linear_model)

dw = (1 / n_samples) * np.dot(X.T, (________ - y))
db = (1 / n_samples) * np.sum(________ - y)
```

4. Why does `predict` return class labels, while the sigmoid output itself gives probabilities?
