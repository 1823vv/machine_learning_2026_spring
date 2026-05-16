## Question: Predict Function and Logits Versus Probabilities

In our own NumPy neural-network implementation, prediction first computes logits, then applies softmax, and finally chooses the class with the largest probability:

```python
def predict(network, X):
    logits = forward(network, X)[-1]
    probs = softmax(logits)
    return np.argmax(probs, axis=-1)
```

1. Explain what `forward(network, X)[-1]` returns for the MNIST network, including its shape for a batch of `B` images.
2. Write the softmax formula for one row of logits. Explain why softmax probabilities are easier to interpret than logits.
3. Explain why `np.argmax(probs, axis=-1)` returns one predicted digit per input image.
4. If the logits for one image are `[1.0, 4.0, 2.0]` in a three-class toy problem, identify the predicted class without computing exact softmax values. Explain why applying softmax does not change the argmax class.
