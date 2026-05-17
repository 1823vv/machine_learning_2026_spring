## Question: Numerically Stable Softmax

In our own NumPy neural-network implementation, softmax is computed using:

```python
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
```

1. Why can directly computing `np.exp(logits)` be dangerous?
2. What does `np.max(logits, axis=1, keepdims=True)` compute? Why do we subtract the maximum logit separately for each row/sample?
3. Explain mathematically why subtracting the same constant from every logit in a row does not change softmax probabilities. 
4. Give an example of logits where direct exponentiation might overflow. Draw a before/after picture of a row of logits before and after subtracting the maximum.
