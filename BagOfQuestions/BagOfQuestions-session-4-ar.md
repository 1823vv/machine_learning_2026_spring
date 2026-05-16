## Question: Debugging Shape Errors

A student receives this error while running a dense layer:

```text
ValueError: shapes (128,784) and (64,784) not aligned
```

1. What matrix multiplication was probably attempted?
2. What should the weight shape be for an input with 784 features and 64 output units? Why does row-vector convention require weights of shape `(input_units, output_units)`?
3. Draw the correct shape multiplication. Draw the incorrect shape multiplication and mark why it fails.
4. If the output of the first dense layer is `(128, 64)`, what input shape should the next dense layer expect? What `Dense(...)` constructor should be used for the next layer if it has 32 output units?

## Question: Bias Broadcasting Bug

1. What should the bias shape be for a dense layer with 64 output units?
2. Why would a bias shape of `(128,)` be wrong for this layer?
