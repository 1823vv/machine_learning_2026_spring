## Question: Numerically Stable Softmax

In `code-my_nn.py`, softmax is computed using:

```python
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
```

1. Why can directly computing `np.exp(logits)` be dangerous?
2. What does `np.max(logits, axis=1, keepdims=True)` compute?
3. Why do we subtract the maximum logit separately for each row/sample?
4. Explain mathematically why subtracting the same constant from every logit in a row does not change softmax probabilities.
5. Why is `keepdims=True` useful for broadcasting?
6. Give an example of logits where direct exponentiation might overflow.
7. Draw a before/after picture of a row of logits before and after subtracting the maximum.

## Question: Epsilon in Cross-Entropy

The loss uses:

```python
np.log(softmax_probs + 1e-9)
```

1. Why do we add `1e-9` inside the log?
2. What numerical problem can happen if a predicted probability is exactly 0?
3. Does adding `1e-9` change the mathematical idea of cross-entropy, or is it mainly an engineering trick?
