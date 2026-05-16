## Question: Design a Generalization Strategy for MNIST

You train a neural network for MNIST digit recognition. Each image is flattened from `28 x 28` pixels, and the output has 10 classes. After many epochs, training accuracy is very high, but validation accuracy is much lower.

Design a strategy to improve generalization.

1. Explain the symptom using the words training performance, validation performance, and overfitting.
2. Propose one dropout choice. State where you would put dropout and what dropout probability you might try first.
3. Propose one L2 regularization strength search range.
4. Propose one data-augmentation transformation that makes sense for MNIST.
5. Propose one early-stopping rule with patience.
6. Propose a random-search plan for at least three hyperparameters.
7. Draw a workflow diagram showing: choose hyperparameters, train, monitor validation, select best model, final test once.
8. Explain why the final test set should be used only after model selection is finished.

## Question: Parameter Counting with Regularization Layers

Consider a fully connected MNIST network with row-vector inputs:

```text
input 784 -> hidden 64 -> hidden 32 -> output 10
```

Each dense layer has a weight matrix and a bias vector.

1. How many weights are in the first dense layer?
2. How many biases are in the first dense layer?
3. How many weights and biases are in the second dense layer?
4. How many weights and biases are in the output layer?
5. How many trainable parameters are there in total before adding dropout or batch normalization?
6. If dropout is inserted between layers, does dropout add trainable parameters? Explain.
7. If batch normalization is inserted after the first dense layer with 64 hidden units, how many trainable parameters does batch normalization add if it learns one $\gamma$ and one $\beta$ per hidden unit?
