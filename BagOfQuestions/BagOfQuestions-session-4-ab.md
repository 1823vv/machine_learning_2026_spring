## Question: Logits, Softmax, and Probabilities

A neural network outputs logits for one handwritten-digit image. To keep the arithmetic small, use this three-class toy vector:

```text
z = [2.0, 1.0, 0.0]
```

1. Explain what logits are and why they are not yet probabilities.
2. Write the softmax formula for class $k$, and explain why the softmax outputs can be interpreted as probabilities that sum to 1.
3. Without calculating exact exponentials, which class has the largest probability? If we add the same constant 100 to every logit, does the softmax probability vector change? Explain.
4. Draw a schema: `hidden representation -> logits -> softmax -> probabilities -> predicted class`.

## Question: Prediction Rule

In a prediction helper for a multiclass neural network, the predicted class is selected with `np.argmax(probs, axis=-1)`.

1. Explain what `argmax` returns for each row of probabilities.
2. Explain why selecting the largest probability is consistent with the largest logit before softmax.
3. If two classes have exactly the same probability, what tie issue could happen conceptually?
