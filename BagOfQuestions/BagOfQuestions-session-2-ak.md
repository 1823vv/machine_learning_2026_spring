## Question: Logistic Regression Predict Method

We are implementing binary logistic regression `class MyOwnLogisticRegression` from scratch with NumPy. The model first computes a linear score, then calls its internal `_sigmoid` helper, and finally converts probabilities into class labels.

Fill in the `____YOUR_CODE_HERE__N_____` blanks in the `predict` method below.

```python
def predict(self, X):
    linear_model = np.dot(____YOUR_CODE_HERE__1_____, ____YOUR_CODE_HERE__2_____) + self.bias
    y_predicted = self._sigmoid(____YOUR_CODE_HERE__3_____)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in ____YOUR_CODE_HERE__4_____]
    return np.array(____YOUR_CODE_HERE__5_____)
```

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
- `____YOUR_CODE_HERE__3_____`:
- `____YOUR_CODE_HERE__4_____`:
- `____YOUR_CODE_HERE__5_____`:

Then answer the following short questions:

1. What is the difference between `linear_model`, `y_predicted`, and `y_predicted_cls`?
2. Why does this implementation call `self._sigmoid(...)` rather than a public `sigmoid(...)` method?
3. For probabilities `[0.2, 0.5, 0.8]`, what class labels does the shown threshold rule return? Explain the role of the strict `>` sign.
