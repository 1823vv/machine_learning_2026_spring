## Question: Debug the Dense Layer

A student writes the following incorrect dense layer:

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

1. What is wrong with using `input * self.weights` in the forward pass? What should the correct forward line be?
2. What is wrong with using `self.input.T * grad_output` for `grad_weights`? What should the correct `grad_weights` line be?
3. What is wrong with using `grad_output * self.weights.T` for `grad_input`? What should the correct `grad_input` line be?
4. What is wrong with updating parameters using `+ self.learning_rate * gradient`? Rewrite the corrected version of this `Dense` class.
