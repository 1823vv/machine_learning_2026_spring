## Question: Forward Function and Stored Activations

In our own NumPy neural-network implementation, the forward helper passes a batch through each layer and stores every layer output:

```python
def forward(network, X):
    activations = []
    input = X

    for layer in network:
        input = layer.forward(input)
        activations.append(input)

    return activations
```

For the network `Dense(784, 64) -> ReLU -> Dense(64, 32) -> ReLU -> Dense(32, 10)`, suppose `X.shape == (128, 784)`.

1. Fill in a table with one row per layer showing the shape appended to `activations` after that layer.
2. Explain why the variable named `input` is reassigned after each layer. What does `activations[-1]` represent?
3. Explain why the final output is called logits rather than probabilities. Which separate function converts logits into probabilities?
4. Should this `forward(network, X)` helper update `self.weights` or `self.biases`? Explain.
