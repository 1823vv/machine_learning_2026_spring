# Notation Addendum for Session 7

Session 7 follows the row-vector neural-network notation from `session-0/lecture-0-notation-for-session-1-to-session-7.md`:

$$
z^{(l)} = a^{(l-1)}W^{(l)} + b^{(l)},
\qquad
a^{(l)} = f^{(l)}(z^{(l)}).
$$

For dropout, use activation notation consistently:

| Symbol | Meaning |
|--------|---------|
| $a$ | Activation before dropout for one unit, one vector, or one layer |
| $\tilde a$ | Activation after dropout |
| $p$ | Drop probability |
| $1-p$ | Keep probability |
| $m$ | Binary keep mask: $m_i=1$ means keep unit $i$, $m_i=0$ means drop unit $i$ |
| $\odot$ | Element-wise multiplication |

For one activation coordinate:

$$
m_i \sim \operatorname{Bernoulli}(1-p),
\qquad
\tilde a_i = a_i\frac{m_i}{1-p}.
$$

For a vector or matrix of activations:

$$
\tilde a = a \odot \frac{m}{1-p}.
$$

The expectation calculation should be written coordinate-by-coordinate:

$$
\mathbb{E}[\tilde a_i]
= \mathbb{E}\left[a_i\frac{m_i}{1-p}\right]
= a_i\frac{\mathbb{E}[m_i]}{1-p}
= a_i\frac{1-p}{1-p}
= a_i.
$$

During inference, inverted dropout does nothing:

$$
\tilde a = a.
$$

> Note: the code variable `self.mask` stores the scaled mask $m/(1-p)$. In the math notes, $m$ denotes the binary keep mask before scaling.
