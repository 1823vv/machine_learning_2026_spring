## Question: The `predict(network, X)` Function

The prediction function is:

```python
def predict(network, X):
    logits = forward(network, X)[-1]
    probs = softmax(logits)
    return np.argmax(probs, axis=-1)
```

1. Why does prediction call `forward(network, X)`?
2. Why do we apply softmax to logits before interpreting them as probabilities? What does `axis=-1` mean in `np.argmax(probs, axis=-1)`?
3. If `probs` has shape `(100, 10)`, what is the shape of the returned prediction array? If one row of probabilities is `[0.1, 0.7, 0.2]`, what class is predicted?
4. In this implementation, does `predict` update the model parameters? Draw the prediction pipeline from image pixels to class index.

## Question: Accuracy Calculation

1. Explain this line:

   ```python
   train_accuracy = np.mean(train_predictions == y_train)
   ```

2. If 80 out of 100 predictions are correct, what is the accuracy?
