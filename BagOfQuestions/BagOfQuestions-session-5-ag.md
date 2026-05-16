## Question: Bias Correction in Adam

Adam initializes the moving averages $m$ and $v$ at zero. Early in training, this can bias the moving averages toward zero. Adam therefore uses bias-corrected moments:

$$
\hat m = \frac{m}{1-\beta_1^t},
$$

$$
\hat v = \frac{v}{1-\beta_2^t}.
$$

1. Why are $m$ and $v$ biased toward zero at the beginning of training?
2. What is the meaning of the time step $t$?
3. Explain why the correction factor is especially important for small $t$.
4. What happens to $1-\beta_1^t$ as $t$ becomes large?
5. What happens to $1-\beta_2^t$ as $t$ becomes large?
6. Write the full Adam update rule using $\hat m$, $\hat v$, $\eta$, and $\epsilon$.

## Question: Adam Notation Alert

In optimization, the symbol $v$ can mean different things depending on the algorithm. In momentum, $v$ usually means velocity. In Adam, $v$ usually means a second-moment moving average of squared gradients.

1. In momentum, what does $v$ store?
2. In Adam, what does $v$ store?
3. Why can using the same symbol be confusing?
4. Why is it important to define notation before writing formulas in a written answer?
5. Rewrite one sentence explaining Adam's $v$ without using the word “velocity.”
