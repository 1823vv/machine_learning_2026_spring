## Question: Compare GD, SGD, Momentum, and Adam

Create a comparison table with these columns:

```text
Optimizer | Gradient source | Extra state | Main advantage | Main weakness
```

Fill it for:

1. full-batch GD,
2. mini-batch SGD,
3. Momentum,
4. Adam.

Then answer:

5. Which optimizer uses the whole dataset for each update?
6. Which optimizer introduces mini-batch noise?
7. Which optimizer stores a velocity-like moving average?
8. Which optimizer stores both first and second moments?
9. Which optimizer adapts learning rates per parameter?
10. Draw a concept map showing Adam as “SGD + momentum-like smoothing + adaptive scaling.”

## Question: Choosing an Optimizer

1. Why is GD useful as a teaching baseline?
2. Why is SGD important in modern deep learning?
3. Why is Adam often a strong default choice?
