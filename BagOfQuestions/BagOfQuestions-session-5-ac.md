## Question: Full-Batch Gradient Descent versus Mini-Batch SGD

For a dataset $\mathcal{D}=\{(x^{(i)},y^{(i)})\}_{i=1}^n$, full-batch gradient descent computes a gradient using all $n$ training examples. Mini-batch SGD computes a gradient using a mini-batch $\mathcal{B}$ of size $B$, where $1 < B \ll n$ in the practical mini-batch meaning of SGD.

Full-batch gradient:

$$
g = \frac{1}{n}\sum_{i=1}^{n}\frac{\partial \ell^{(i)}}{\partial W}.
$$

Mini-batch gradient:

$$
g = \frac{1}{B}\sum_{i\in\mathcal{B}}\frac{\partial \ell^{(i)}}{\partial W}.
$$

1. Explain the meaning of $B$ and $\mathcal{B}$. What is the batch size for full-batch gradient descent?
2. What is the batch size for one-sample SGD? What is the usual practical meaning of “SGD” in modern mini-batch training?
3. Draw two optimization paths: a smooth full-batch path and a noisier mini-batch path. Why can full-batch gradient descent be expensive for very large datasets?
4. Why can mini-batch SGD be faster in practice? Why can mini-batch noise sometimes help optimization?

## Question: Batch Size Trade-Off

In mini-batch training, the batch size $B$ affects computation time, gradient noise, memory use, and sometimes generalization.

1. What usually happens when the batch size is very small?
2. What usually happens when the batch size is very large? Why does batch size interact with learning rate?
3. Give one reason a practitioner might choose a medium-sized mini-batch instead of using the entire dataset.
4. Draw a simple comparison of noisy small-batch updates and smoother large-batch updates.
