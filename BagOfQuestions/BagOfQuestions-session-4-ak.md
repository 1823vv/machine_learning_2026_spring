## Question: The `forward(network, X)` Function

A minimal NumPy neural-network implementation uses:

```python
def forward(network, X):
    activations = []
    input = X

    for layer in network:
        input = layer.forward(input)
        activations.append(input)

    return activations
```

1. What does this function return?
2. If the network has 5 layers in the list, how many activation objects are returned?
3. Why is `input` reassigned after each layer?
4. What is `activations[-1]`?
5. Why is the final output called logits instead of probabilities?
6. Draw a table with one row per layer for the MNIST network and include the output shape of each layer for batch size 128.
7. Should `forward(network, X)` modify weights? Explain.
