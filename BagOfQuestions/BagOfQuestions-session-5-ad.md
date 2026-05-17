## Question: Bias Correction in Adam

Adam initializes the moving averages $m$ and $v$ at zero. Early in training, this can bias the moving averages toward zero. Adam therefore uses bias-corrected moments:

$$
\hat m = \frac{m}{1-\beta_1^t},
$$

$$
\hat v = \frac{v}{1-\beta_2^t}.
$$

1. Why are $m$ and $v$ biased toward zero at the beginning of training?
2. What is the meaning of the time step $t$? Explain why the correction factor is especially important for small $t$.
3. What happens to $1-\beta_1^t$ as $t$ becomes large? What happens to $1-\beta_2^t$ as $t$ becomes large?
4. Write the full Adam update rule using $\hat m$, $\hat v$, $\eta$, and $\epsilon$.
