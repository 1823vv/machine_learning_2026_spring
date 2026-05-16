## Question: Dropout as Training Many Sub-Networks

Dropout randomly removes some activations during training. In a neural network, this means each mini-batch may effectively train a slightly different sub-network.

1. Explain in words what dropout does during training.
2. Draw a neural network layer before and after dropout is applied.
3. Why can dropout reduce co-adaptation between neurons?
4. Why can dropout improve generalization?
5. What does dropout do during evaluation/inference?
6. Why should dropout behavior be different during training and evaluation?

## Question: Dropout and Generalization

Suppose a neural network overfits: training accuracy is high, but validation accuracy is much lower. A practitioner adds dropout between hidden layers.

1. What dropout probability $p$ might be reasonable to try first for hidden layers?
2. What can happen if $p$ is too small?
3. What can happen if $p$ is too large?
4. Draw training and validation curves before and after a successful dropout change.
5. Explain why the best dropout probability should be selected using validation data.
