## Question: Inverted Dropout Formula and Expectation

In inverted dropout, each activation is multiplied by a random mask during training. For dropout probability $p$, the keep probability is $1-p$:

$$
\text{mask} \sim \frac{\mathrm{Bernoulli}(1-p)}{1-p}.
$$

The dropout output for an activation $a$ is

$$
\tilde a = a \cdot \text{mask}.
$$

1. What is the probability that a unit is kept?
2. What is the probability that a unit is dropped?
3. Why do we divide by $1-p$ in inverted dropout?
4. Show that the expected value of $\tilde a$ equals $a$.
5. What happens during inference when using inverted dropout?
6. Draw a small network showing dropped units and scaled surviving units.

## Question: Original Dropout versus Inverted Dropout

There are two common ways to handle dropout scaling: scale activations at training time, or scale them at inference time.

1. In inverted dropout, when are surviving activations scaled?
2. In original dropout, when are activations scaled?
3. Why is inverted dropout convenient at inference time?
4. What would go wrong if we dropped units during inference for normal prediction?
5. Explain the phrase “training mode versus evaluation mode” for dropout.
