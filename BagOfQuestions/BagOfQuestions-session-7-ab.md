## Question: Dropout as Training Many Sub-Networks

Dropout randomly removes some activations during training. With drop probability $p$, each mini-batch may effectively train a slightly different sub-network.

1. Draw a neural network layer before and after dropout is applied.
2. Explain in words what dropout does during training. Why can dropout reduce co-adaptation between neurons, and why can this improve generalization?
3. What does dropout do during evaluation/inference?
4. Why should dropout behavior be different during training and evaluation?

## Question: Dropout and Generalization

Suppose a neural network overfits: training accuracy is high, but validation accuracy is much lower. A practitioner adds dropout between hidden layers, using $p$ as the drop probability.

1. What dropout probability $p$ might be reasonable to try first for hidden layers?
2. What can happen if $p$ is too small? What can happen if $p$ is too large?
3. Draw training and validation curves before and after a successful dropout change.
4. Explain why the best dropout probability should be selected using validation data.
