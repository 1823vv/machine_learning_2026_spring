## Question: Mapping Session 4 Code to PyTorch Ideas

Session 4 does not yet use explicit `model`, `criterion`, and `optimizer` objects, but the roles are already present.

1. Which function in Session 4 plays a role similar to `model(X)`?
2. Which function plays a role similar to `criterion(logits, y)`?
3. Which part of the code plays a role similar to `loss.backward()`?
4. Which part of the code plays a role similar to `optimizer.step()`?
5. Draw a two-column table comparing Session 4 code with a PyTorch-style training step.
6. Why is it pedagogically useful to first implement these ideas manually with NumPy?
7. What design improvement happens in later sessions when optimization is separated from `Dense.backward()`?

## Question: Responsibility of `Dense.backward`

In Session 4, `Dense.backward()` both computes gradients and updates parameters.

1. Why is this simple for a first implementation?
2. Why might this design become limiting when adding Momentum or Adam later?
