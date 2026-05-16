## Question: Debug the Dense Layer

In our own NumPy neural-network implementation, a dense layer should use row-vector batches. For a batch input `input` with shape `(B, input_units)`, `self.weights` has shape `(input_units, output_units)`, `self.biases` has shape `(output_units,)`, and `grad_output` has shape `(B, output_units)`. A student writes the following incorrect dense layer:

```python
class Dense(Layer):
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

1. Explain why `input * self.weights` is not the correct affine transformation for a dense layer. Write the corrected `forward` line, preserving `self.input = input`.
2. Explain why `self.input.T * grad_output` and `grad_output * self.weights.T` are wrong for the two backward matrix multiplications. Write the corrected `grad_weights` and `grad_input` lines.
3. Explain the correct direction of the gradient-descent update. Rewrite the corrected updates for `self.weights` and `self.biases`.
4. For `B=128`, `input_units=64`, and `output_units=32`, check the shapes of `grad_weights`, `grad_biases`, and `grad_input`.
