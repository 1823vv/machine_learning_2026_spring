## Question: The Layer Interface

In our own NumPy neural-network implementation, all layers follow a simple interface:

```python
class Layer:
    def forward(self, input):
        return input

    def backward(self, grad_output):
        return grad_output
```

The MNIST network is `Dense(784, 64) -> ReLU -> Dense(64, 32) -> ReLU -> Dense(32, 10)`.

1. Explain why every layer needs a `forward` method and a `backward` method, even though only dense layers have trainable parameters.
2. Identify which layers in the MNIST network store trainable parameters and which layers only transform activations or gradients.
3. Draw a sequence diagram for `forward(network, X)` from the first dense layer to the final logits. Then draw the reverse order used during training.
4. Explain how the shared interface makes it possible for the training loop to call `layer.backward(grad_output)` without special-case code for each layer type.
