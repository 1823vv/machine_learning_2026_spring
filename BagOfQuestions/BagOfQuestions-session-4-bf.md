## Question: Dense Layer Backward Fill-in-the-Blank

In our own NumPy neural-network implementation, a dense layer stores the incoming batch in `self.input` during the forward pass. During backpropagation, `grad_output` is the gradient of the loss with respect to the dense layer output.

Fill in the `____YOUR_CODE_HERE__N_____` blanks.

```python
class Dense(Layer):
    def backward(self, grad_output):
        grad_weights = np.dot(____YOUR_CODE_HERE__1_____, ____YOUR_CODE_HERE__2_____)
        grad_biases = np.sum(____YOUR_CODE_HERE__3_____, axis=0)
        grad_input = np.dot(____YOUR_CODE_HERE__4_____, ____YOUR_CODE_HERE__5_____)

        self.weights = self.weights - self.learning_rate * ____YOUR_CODE_HERE__6_____
        self.biases = self.biases - self.learning_rate * ____YOUR_CODE_HERE__7_____

        return ____YOUR_CODE_HERE__8_____
```

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
- `____YOUR_CODE_HERE__3_____`:
- `____YOUR_CODE_HERE__4_____`:
- `____YOUR_CODE_HERE__5_____`:
- `____YOUR_CODE_HERE__6_____`:
- `____YOUR_CODE_HERE__7_____`:
- `____YOUR_CODE_HERE__8_____`:

1. If `self.input.shape == (B, d_in)` and `grad_output.shape == (B, d_out)`, state the shapes of `grad_weights`, `grad_biases`, and `grad_input`.
2. Explain why `grad_input` must be returned for the previous layer.
