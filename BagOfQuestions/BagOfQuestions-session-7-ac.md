## Question: Inverted Dropout Formula and Expectation

In our own NumPy neural-network implementation, inverted dropout randomly drops activations during training and scales the surviving activations so that no extra scaling is needed during inference. Let $p$ be the drop probability. For one activation coordinate $a_i$, sample a binary keep mask

$$
m_i \sim \operatorname{Bernoulli}(1-p),
$$

where $m_i=1$ means the activation is kept and $m_i=0$ means it is dropped. The inverted-dropout output is

$$
\tilde a_i = a_i\frac{m_i}{1-p}.
$$

1. Draw a small neural network layer before and after dropout, showing which units are kept and which units are dropped.
2. Explain why inverted dropout divides by $1-p$ during training.
3. Show the expectation calculation proving that $\mathbb{E}[\tilde a_i]=a_i$.
4. State what happens to the dropout layer during evaluation/inference.
