## Question: Softmax Cross-Entropy Function

In our own NumPy neural-network implementation, the loss function receives `logits` with shape `(B, K)` and integer class labels `labels` with shape `(B,)`. It creates one-hot labels, computes numerically stable softmax probabilities, returns the average cross-entropy loss, and returns the gradient with respect to logits.

Fill in the `____YOUR_CODE_HERE__N_____` blanks.

```python
def softmax_crossentropy_with_logits(logits, labels):
    batch_size = logits.shape[0]
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    loss = -np.sum(____YOUR_CODE_HERE__1_____ * np.log(____YOUR_CODE_HERE__2_____ + 1e-9)) / batch_size
    grad = (____YOUR_CODE_HERE__3_____ - ____YOUR_CODE_HERE__4_____) / ____YOUR_CODE_HERE__5_____

    return ____YOUR_CODE_HERE__6_____, ____YOUR_CODE_HERE__7_____
```

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
- `____YOUR_CODE_HERE__3_____`:
- `____YOUR_CODE_HERE__4_____`:
- `____YOUR_CODE_HERE__5_____`:
- `____YOUR_CODE_HERE__6_____`:
- `____YOUR_CODE_HERE__7_____`:

1. Explain why the code subtracts `np.max(logits, axis=1, keepdims=True)` before exponentiation.
2. For one row of `logits`, explain why `softmax_probs - one_hot_labels` has positive entries for classes whose probability should decrease and a negative entry for the true class when the true-class probability is too small.
