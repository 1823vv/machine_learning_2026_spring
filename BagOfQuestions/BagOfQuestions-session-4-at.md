## Question: Integrated Code Path — From Pixels to One Update

Explain one NumPy training epoch on MNIST.

Your answer should use these exact terms:

- normalized pixels,
- batch matrix $X$,
- dense layer,
- ReLU,
- logits,
- softmax probabilities,
- cross-entropy loss,
- reverse layer loop,
- gradient descent update,
- training accuracy,
- validation accuracy.

Then answer:

1. Where in the code are pixels normalized?
2. Where in the code are logits produced? Where in the code is cross-entropy computed?
3. Where in the code are predictions converted into class indices? Where in the code are training and validation accuracy printed?
4. Draw a single large schema connecting all parts of the training path. If validation accuracy is much lower than training accuracy, what problem might be happening?

## Question: Short Final Check

1. Write the formula for softmax.
2. Write the formula for multi-class cross-entropy.
3. Write the formula for ReLU.
4. State the total number of trainable parameters in the MNIST network.
