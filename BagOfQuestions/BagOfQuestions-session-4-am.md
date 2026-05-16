## Question: He Initialization

In the `Dense` layer, the weights are initialized by:

```python
self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
```

1. What is the shape of `self.weights`?
2. What distribution does `np.random.randn(...)` sample from? What is the purpose of multiplying by `np.sqrt(2.0 / input_units)`?
3. Why is this initialization especially connected to ReLU-like activations? The formula is often described as a normal distribution centered at 0. Where is the 0 in the code?
4. For `Dense(784, 64)`, what is the scaling factor? Draw a small histogram-style sketch of randomly initialized weights centered near 0.

## Question: Bad Initialization

1. What could go wrong if all weights are initialized to exactly 0?
2. What could go wrong if weights are initialized with values that are too large?
