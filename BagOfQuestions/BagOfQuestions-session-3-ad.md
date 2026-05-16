## Question: Output Layers Depend on the Task

A neural network's final layer must match the prediction task. Assume that the last hidden representation of one example is a vector $h$. The network then produces logits or outputs using a final affine transformation such as

$$
z = Wh + b.
$$

For each case below, write the appropriate output dimension, the usual final activation, and the typical loss formula.

1. **One-output regression.** The target is one continuous number $y \in \mathbb{R}$. What should the output dimension be? Should the final output usually be linear, sigmoid, or softmax? Write the mean squared error loss for one example.
2. **Binary classification.** The target is $y \in \{0,1\}$. What should the output dimension be if we use one logit? Write the sigmoid formula $\sigma(z)$. Write the binary cross-entropy loss for one example.
3. **Multiclass classification with 10 classes.** The target class is $c \in \{0,1,\ldots,9\}$. What should the output dimension be? Write the softmax formula for class $k$. Write the cross-entropy loss for one example using the correct class $c$.
4. In words, explain why the output layer and the loss function should be designed together.
5. For MNIST digit classification, what does each output neuron represent before softmax, and what does each output value represent after softmax?
