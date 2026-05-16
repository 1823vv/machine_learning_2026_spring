## Question: Fill in the Training Function

In our own NumPy neural-network implementation, one training step runs a forward pass, computes softmax cross-entropy, and then walks backward through the layer list. Fill in the `____YOUR_CODE_HERE__N____` blanks.

```python
def train(network, X, y):
    activations = ____YOUR_CODE_HERE__1____
    logits = ____YOUR_CODE_HERE__2____

    loss, grad_logits = ____YOUR_CODE_HERE__3____(logits, y)

    grad_output = ____YOUR_CODE_HERE__4____
    for i in range(len(network))[____YOUR_CODE_HERE__5____]:
        layer = network[i]
        grad_output = layer.____YOUR_CODE_HERE__6____(grad_output)

    return ____YOUR_CODE_HERE__7____
```

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1____`:
- `____YOUR_CODE_HERE__2____`:
- `____YOUR_CODE_HERE__3____`:
- `____YOUR_CODE_HERE__4____`:
- `____YOUR_CODE_HERE__5____`:
- `____YOUR_CODE_HERE__6____`:
- `____YOUR_CODE_HERE__7____`:

## Question: Explain One Training Step

Explain one training step in words using this order: `X -> forward -> logits -> loss -> initial gradient -> reverse layer loop -> parameter updates`.
