## Question: Neural Network Figure with Code Names

Draw the MNIST network and annotate it with the exact code class names.

Your figure should include:

- `Dense(784, 64)`,
- `ReLU()`,
- `Dense(64, 32)`,
- `ReLU()`,
- `Dense(32, 10)`,
- `softmax_crossentropy_with_logits`,
- predicted probabilities,
- predicted class using `argmax`.

Then answer:

1. Which boxes are layers in the `network` list?
2. Which boxes are not layers in the `network` list but are still needed for loss or prediction? Which boxes contain trainable parameters?
3. Which boxes are activation functions? Which box produces logits?
4. Which box produces probabilities? Why is it useful to draw softmax separately from the final dense layer?
