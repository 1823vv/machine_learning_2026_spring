## Question: The PyTorch Training Rhythm Without Code

Session 5 connects our from-scratch neural network to the PyTorch mental model:

```text
model(X_batch) -> loss -> zero_grad -> backward -> step
```

Explain each stage in words.

1. What happens during `model(X_batch)`?
2. What does the loss compare?
3. Why do old gradients need to be cleared before computing new gradients?
4. What happens during the backward step?
5. What happens during the optimizer step?
6. Draw this pipeline as a loop over mini-batches.
7. In this rhythm, where are parameters changed?
8. In this rhythm, where are gradients computed?

## Question: Responsibility Separation

Create a table with three rows:

```text
Component | Responsibility | Examples
```

Fill it for:

1. layers,
2. loss function,
3. optimizer.

Then explain why mixing these responsibilities makes the code harder to extend.
