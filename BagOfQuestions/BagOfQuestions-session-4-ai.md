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
2. Explain why it is useful for every layer to have a `backward` method.
3. Which layers in the MNIST network have trainable parameters?
4. Which layers do not have trainable parameters?
5. Draw a sequence diagram showing how `forward(network, X)` calls each layer.
6. Draw a second sequence diagram showing the reverse order used during training.
7. Explain why this simple interface makes it easier to add new layers later.

## Question: Layer List Design

1. Why is the network represented as a Python list of layers?
2. What is one advantage of this design compared with hard-coding every layer separately?
