## Question: Mapping NumPy Training Code to PyTorch Ideas

The NumPy implementation does not use explicit `model`, `criterion`, and `optimizer` objects, but the roles are already present.

1. Which function plays a role similar to `model(X)`?
2. Which function plays a role similar to `criterion(logits, y)`?
3. Which part of the code plays a role similar to `loss.backward()`?
4. Which part of the code plays a role similar to `optimizer.step()`?
5. Draw a two-column table comparing the NumPy code with a PyTorch-style training step.
6. Why is it pedagogically useful to first implement these ideas manually with NumPy?
7. What design improvement happens in a more modular implementation when optimization is separated from `Dense.backward()`?

## Question: Responsibility of `Dense.backward`

In this simple implementation, `Dense.backward()` both computes gradients and updates parameters.

1. Why is this simple for a first implementation?
2. Why might this design become limiting when adding Momentum or Adam later?
