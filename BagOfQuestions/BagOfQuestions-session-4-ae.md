## Question: Dense Layer Shapes

In the course implementation, a dense layer uses:

```python
return np.dot(input, self.weights) + self.biases
```

Assume:

```text
input.shape = (128, 784)
self.weights.shape = (784, 64)
self.biases.shape = (64,)
```

1. What is the batch size?
2. What is `input_units`?
3. What is `output_units`?
4. What is the output shape of this dense layer?
5. Why does `np.dot(input, self.weights)` work here?
6. Why can `self.biases` with shape `(64,)` be added to a matrix with shape `(128, 64)`?
7. Draw this dense layer as a matrix multiplication diagram.

## Question: The Meaning of `self.input`

1. In `Dense.forward`, why does the layer store `self.input = input`?
2. Is `self.input` a trainable parameter?
3. Is `self.input` one sample or a batch in the training code?
