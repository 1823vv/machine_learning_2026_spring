## Question: The PyTorch Training Rhythm Without Writing Code

In a typical neural-network training loop, the same conceptual rhythm appears in frameworks such as PyTorch:

```text
model(X_batch) -> loss -> zero_grad -> backward -> step
```

Here $X_{batch}$ is a mini-batch of inputs, the model produces predictions, the loss compares predictions to targets, gradients are computed, and the optimizer updates parameters.

1. What happens during `model(X_batch)`? What does the loss compare?
2. Why do old gradients need to be cleared before computing new gradients? What happens during the backward step?
3. What happens during the optimizer step? Draw this pipeline as a loop over mini-batches.
4. In this rhythm, where are parameters changed? In this rhythm, where are gradients computed?

## Question: Responsibility Separation in a Training Loop

A clean neural-network implementation separates the responsibilities of layers, the loss function, and the optimizer. This makes the training loop easier to debug and extend.

Create a table with these columns:

```text
Component | Responsibility | Examples
```

Fill the table for:

1. layers,
2. loss function,
3. optimizer.

Then answer:

4. Why does mixing layer logic, loss logic, and optimizer logic make code harder to extend?
5. In which component should the parameter update $W \leftarrow W - \eta g$ belong?
