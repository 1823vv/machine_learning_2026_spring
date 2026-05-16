## Question: MNIST Network Architecture and Parameter Counting

In our own Neural Network From Scratch implementation with NumPy for MNIST digit recognition, each grayscale image has shape `28 x 28` and is flattened into one row vector $x \in \mathbb{R}^{1 \times 784}$. The network architecture is:

```text
Dense(784, 64) -> ReLU -> Dense(64, 32) -> ReLU -> Dense(32, 10)
```

Use the row-vector convention: for `Dense(d_in, d_out)`, the weight matrix has shape $W \in \mathbb{R}^{d_{in} \times d_{out}}$ and the bias has shape $b \in \mathbb{R}^{1 \times d_{out}}$.

1. Draw this neural network as a schema with input layer, hidden layers, ReLU activations, logits layer, and softmax probability output.
2. Why is the input dimension 784 for MNIST digit recognition?
3. Why is the final output dimension 10?
4. For `Dense(784, 64)`, what is the shape of $W^{(1)}$, and how many weight parameters does it contain?
5. For `Dense(784, 64)`, what is the shape of $b^{(1)}$, and how many bias parameters does it contain?
6. For `Dense(64, 32)`, what are the shapes of $W^{(2)}$ and $b^{(2)}$, and how many weights and biases does this layer contain?
7. For `Dense(32, 10)`, what are the shapes of $W^{(3)}$ and $b^{(3)}$, and how many weights and biases does this layer contain?
8. How many trainable parameters are there in total?
9. Does ReLU add trainable parameters in this implementation? Explain.
10. In your drawing, clearly separate the logits $z^{(3)}$ from the softmax probabilities $\hat{Y}$.
