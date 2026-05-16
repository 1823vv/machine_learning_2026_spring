## Question: The Layer Interface

In a minimal NumPy neural-network implementation, all layers follow a simple interface:

```python
class Layer:
    def forward(self, input):
        return input

    def backward(self, grad_output):
        return grad_output
```

1. Explain why it is useful for every layer to have a `forward` method.
2. Explain why it is useful for every layer to have a `backward` method. Which layers in the MNIST network have trainable parameters?
3. Which layers do not have trainable parameters? Draw a sequence diagram showing how `forward(network, X)` calls each layer.
4. Draw a second sequence diagram showing the reverse order used during training. Explain why this simple interface makes it easier to add new layers later.
