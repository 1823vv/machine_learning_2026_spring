## Question: PyTorch Training Pipeline as a Neural Network Pattern

Session 3 shows that linear regression and logistic regression can be written with the same deep-learning training pipeline.

Draw the pipeline:

```text
model -> predictions -> loss -> zero gradients -> backward -> optimizer step
```

Then answer:

1. In PyTorch, what is the role of `nn.Linear` for linear regression?
2. In PyTorch, what extra output activation idea is needed for logistic regression?
3. What is the role of the loss function?
4. What is the role of the optimizer?
5. Why do we reset gradients before computing new gradients?
6. Why is this same pipeline useful for training deeper neural networks later?
7. Compare scikit-learn and PyTorch: which one hides more of the training procedure, and which one exposes it more?

## Question: Linear and Logistic Regression as Neural Networks

1. Draw linear regression as a one-layer neural network.
2. Draw logistic regression as a one-layer neural network plus sigmoid.
3. Explain why this viewpoint is useful before learning deep neural networks.
