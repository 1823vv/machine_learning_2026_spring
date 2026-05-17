## Question: Output Layers Depend on the Task

A neural network's final layer must match the prediction task. Under the row-vector convention, assume the last hidden representation for one example is $a^{(L-1)}$, and the final affine transformation is

$$
z^{(L)} = a^{(L-1)}W^{(L)} + b^{(L)}.
$$

For each case below, write the appropriate output dimension, the usual final activation, and the typical loss formula.

1. **One-output regression.** The target is one continuous number $y \in \mathbb{R}$. What output dimension and final activation should be used? Write the mean squared error loss for one example.
2. **Binary classification.** The target is $y \in \{0,1\}$. If we use one logit, what output dimension and activation should be used? Write the sigmoid formula $\sigma(z)$ and the binary cross-entropy loss for one example.
3. **Multiclass classification with 10 classes.** The target class is $c \in \{0,1,\ldots,9\}$. What output dimension and activation should be used? Write the softmax formula for class $k$ and the cross-entropy loss for one example.
