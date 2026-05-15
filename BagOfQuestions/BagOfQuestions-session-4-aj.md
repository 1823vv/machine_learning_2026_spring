## Question: Fill in the Training Function

Fill in the blanks in this Session 4 training function:

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

Your answers:

- `____YOUR_CODE_HERE__1____`:
- `____YOUR_CODE_HERE__2____`:
- `____YOUR_CODE_HERE__3____`:
- `____YOUR_CODE_HERE__4____`:
- `____YOUR_CODE_HERE__5____`:
- `____YOUR_CODE_HERE__6____`:
- `____YOUR_CODE_HERE__7____`:

## Question: Explain the Story of One Training Step

Explain the training step in words using this order:

```text
X -> forward -> logits -> loss -> initial gradient -> reverse layer loop -> parameter updates
```
