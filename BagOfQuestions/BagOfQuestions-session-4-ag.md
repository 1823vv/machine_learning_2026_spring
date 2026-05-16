## Question: ReLU Layer Fill-in-the-Blank

In our own NumPy neural-network implementation, a ReLU activation layer has no trainable parameters. Its forward pass stores the incoming array in `self.input`, and its backward pass multiplies the upstream gradient by the local ReLU derivative.

Fill in the `____YOUR_CODE_HERE__N_____` blanks.

```python
class ReLU(Layer):
    def forward(self, input):
        self.input = ____YOUR_CODE_HERE__1_____
        return np.maximum(____YOUR_CODE_HERE__2_____, input)

    def backward(self, grad_output):
        relu_grad = ____YOUR_CODE_HERE__3_____ > 0
        return ____YOUR_CODE_HERE__4_____ * relu_grad
```

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
- `____YOUR_CODE_HERE__3_____`:
- `____YOUR_CODE_HERE__4_____`:

1. Explain why `self.input` is needed in `backward` even though ReLU has no trainable parameters.
2. For `self.input = [[-2, 0, 3]]` and `grad_output = [[5, 5, 5]]`, compute the returned gradient.
