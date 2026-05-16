## Question: Softmax Cross-Entropy Code Reading

Read this function call:

```python
loss, grad_logits = softmax_crossentropy_with_logits(logits, y)
```

1. What are `logits`?
2. What is `y`? What scalar does `loss` represent?
3. What does `grad_logits` represent in the training loop? Why does the function combine softmax and cross-entropy instead of requiring a separate softmax layer at the end of the network?
4. What is the shape of `grad_logits` if `logits.shape == (256, 10)`? Draw a diagram showing how logits and labels enter the loss function.

## Question: Last Dense Layer

1. Why does the network end with `Dense(32, 10)` and not `ReLU()`?
2. Why do we not put a softmax layer directly in the network list in this implementation?
