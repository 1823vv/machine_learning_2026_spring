## Question: Logits, Softmax, and Probabilities

A neural network outputs logits for one MNIST image:

```text
z = [2.0, 1.0, 0.0]
```

Assume this is a three-class toy example.

1. What are logits?
2. Are logits probabilities? Explain.
3. Write the softmax formula for class $k$.
4. Explain why softmax outputs are probabilities.
5. Explain why the softmax probabilities must sum to 1.
6. Without calculating exact exponentials, which class has the largest probability?
7. If we add the same constant 100 to every logit, does the softmax probability vector change? Explain.
8. Draw a schema:

   ```text
   hidden representation -> logits -> softmax -> probabilities -> predicted class
   ```

## Question: Prediction Rule

1. In `predict(network, X)`, why do we use `np.argmax(probs, axis=-1)`?
2. If two classes have exactly the same probability, what issue could happen conceptually?
