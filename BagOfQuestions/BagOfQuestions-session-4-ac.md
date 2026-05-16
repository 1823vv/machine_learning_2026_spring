## Question: Numerically Stable Softmax

In our own NumPy neural-network implementation, softmax is computed using:

```python
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
```

1. Why can directly computing `np.exp(logits)` be dangerous?
2. What does `np.max(logits, axis=1, keepdims=True)` compute? Why do we subtract the maximum logit separately for each row/sample?
3. Explain mathematically why subtracting the same constant from every logit in a row does not change softmax probabilities. Why is `keepdims=True` useful for broadcasting?
4. Give an example of logits where direct exponentiation might overflow. Draw a before/after picture of a row of logits before and after subtracting the maximum.

## Question: Epsilon in Cross-Entropy

The loss uses:

```python
np.log(softmax_probs + 1e-9)
```

1. Why do we add `1e-9` inside the log?
2. What numerical problem can happen if a predicted probability is exactly 0?
3. Does adding `1e-9` change the mathematical idea of cross-entropy, or is it mainly an engineering trick?
