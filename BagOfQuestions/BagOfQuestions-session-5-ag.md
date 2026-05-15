## Question: Bias Correction in Adam

Adam initializes the first and second moments at zero. This creates a cold-start issue.

The bias correction formulas are:

$$
\hat{m}^{(t)}=\frac{m^{(t)}}{1-\beta_1^t}
$$

$$
\hat{v}^{(t)}=\frac{v^{(t)}}{1-\beta_2^t}
$$

1. Why are the moving averages biased toward zero at the beginning of training?
2. If $\beta_1=0.9$ and $t=1$, what is $1-\beta_1^t$?
3. Why does dividing by this quantity correct the scale of $m^{(1)}$?
4. As $t$ becomes large, what happens to $\beta_1^t$ and $\beta_2^t$?
5. Why does the correction naturally become less important later in training?
6. Draw a curve showing a correction factor starting large and approaching 1 over time.
7. Explain why early updates might be too small without bias correction.

## Question: Notation Alert

1. In $\beta_1^t$, what does the superscript $t$ mean?
2. Why is it important not to confuse this with layer index notation from neural networks?
