## Question: Training Step Fill-in-the-Blank

In our own NumPy neural-network implementation, one training step runs a forward pass, computes softmax cross-entropy, and then walks backward through the layer list in reverse order.

Fill in the `____YOUR_CODE_HERE__N_____` blanks.

```python
def train(network, X, y):
    activations = ____YOUR_CODE_HERE__1_____(network, X)
    logits = activations[____YOUR_CODE_HERE__2_____]

    loss, grad_logits = ____YOUR_CODE_HERE__3_____(logits, y)

    grad_output = ____YOUR_CODE_HERE__4_____
    for i in range(len(network))[____YOUR_CODE_HERE__5_____]:
        layer = network[i]
        grad_output = layer.____YOUR_CODE_HERE__6_____(grad_output)

    return ____YOUR_CODE_HERE__7_____
```

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
- `____YOUR_CODE_HERE__3_____`:
- `____YOUR_CODE_HERE__4_____`:
- `____YOUR_CODE_HERE__5_____`:
- `____YOUR_CODE_HERE__6_____`:
- `____YOUR_CODE_HERE__7_____`:

1. Explain why the loop must visit layers in reverse order during backpropagation.
2. Explain why `grad_logits` is the correct initial value for `grad_output`.
