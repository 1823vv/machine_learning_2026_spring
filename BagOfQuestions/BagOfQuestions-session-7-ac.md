## Question: Inverted Dropout Formula and Expectation

In inverted dropout, each activation is multiplied by a random mask during training. For dropout probability $p$, the keep probability is $1-p$:

$$
\text{mask} \sim \frac{\mathrm{Bernoulli}(1-p)}{1-p}.
$$

The dropout output for an activation $a$ is

$$
\tilde a = a \cdot \text{mask}.
$$

1. Draw a small network showing dropped units and. 
2. Why do we divide by $1-p$ in inverted dropout?
3. Show that the expected value of $\tilde a$ equals $a$.
4. What happens during inference when using inverted dropout?

