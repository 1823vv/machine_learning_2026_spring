## Question: Replacing ReLU with Sigmoid in NN from Scratch

In our NumPy neural network implementation, we want to replace ReLU with Sigmoid.

1. Mathematically, how does the forward formula change? How does the backward gradient change?

2. Practically, name one potential problem when using Sigmoid in deep networks.

3. Write the `forward` and `backward` methods for a `Sigmoid` class that inherits from `Layer`. Start with `class Sigmoid(Layer):`.

4. Write the same for `class LeakyReLU(Layer):` (use $\alpha = 0.01$).


## Question: Debugging Dense Layer and Network Schema

Consider this incorrect `Dense` class from our NumPy NN implementation:

```python
class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
        self.biases = np.zeros(output_units)

    def forward(self, input):
        self.input = input
        return input * self.weights + self.biases

    def backward(self, grad_output):
        grad_weights = self.input.T * grad_output
        grad_biases = np.sum(grad_output, axis=0)
        grad_input = grad_output * self.weights.T
        self.weights = self.weights + self.learning_rate * grad_weights
        self.biases = self.biases + self.learning_rate * grad_biases
        return grad_input
```

1. Identify **two bugs** in `forward` and **two bugs** in `backward`. Write corrected lines.

2. He initialization uses $W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$. The code shows no explicit zero mean. Why is that correct?

3. Draw a schema of an MNIST network (784 → 64 → 32 → 10) using our implementation. Show: Input, Dense, ReLU, Dense, ReLU, Dense (logits), Softmax activation. Label each layer.


## Question: Training Loop Fill‑in‑the‑Blank

Complete the training function for a neural network.

```python
def train(network, X, y):
    # Forward pass
    activations = ____YOUR_CODE_HERE__1_____(network, X)
    logits = activations____YOUR_CODE_HERE__2_____

    # Loss and initial gradient
    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)

    # Backward pass
    grad_output = grad_logits
    for i in range(len(network))[____YOUR_CODE_HERE__3_____]:
        layer = network[i]
        grad_output = layer.backward(____YOUR_CODE_HERE__4_____)

    return loss
```

Your answers after the `:` (one term or short expression each):

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
- `____YOUR_CODE_HERE__3_____`:
- `____YOUR_CODE_HERE__4_____`: