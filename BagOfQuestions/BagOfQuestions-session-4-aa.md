## Question: MNIST Network Architecture and Parameter Counting

In `session-4/code-my_nn.py`, the MNIST network is:

```text
Dense(784, 64) -> ReLU -> Dense(64, 32) -> ReLU -> Dense(32, 10)
```

1. Draw this neural network as a schema with input layer, hidden layers, ReLU activations, logits layer, and softmax probability output.
2. Why is the input dimension 784 for MNIST?
3. Why is the final output dimension 10?
4. How many weights are in `Dense(784, 64)`?
5. How many biases are in `Dense(784, 64)`?
6. How many weights and biases are in `Dense(64, 32)`?
7. How many weights and biases are in `Dense(32, 10)`?
8. How many trainable parameters are there in total?
9. Does ReLU add trainable parameters in this implementation? Explain.
10. In your drawing, clearly separate logits from probabilities.
