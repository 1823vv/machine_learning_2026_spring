## Question: Activation Layer Variants

In our own NumPy neural-network implementation, an activation layer stores its incoming array as `self.input` during `forward` so that `backward` can compute a local derivative. The existing ReLU layer uses the pattern:

```python
class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output):
        relu_grad = self.input > 0
        return grad_output * relu_grad
```

1. Write the mathematical formulas for sigmoid and Leaky ReLU with slope $\alpha=0.01$ on the negative side. Draw ReLU, sigmoid, and Leaky ReLU on the same axes and label where their gradients are small or zero.
2. Write complete `Sigmoid(Layer)` and `LeakyReLU(Layer)` classes using the same `forward(self, input)` and `backward(self, grad_output)` interface. Your code should store `self.input`, not a differently named variable.
3. Explain one possible advantage of Leaky ReLU compared with ReLU, and one possible disadvantage of sigmoid in deeper networks.
4. If an activation layer has no trainable weights or biases, why does it still need a `backward` method during backpropagation?
