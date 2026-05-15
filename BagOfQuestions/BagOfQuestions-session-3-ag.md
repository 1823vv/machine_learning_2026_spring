## Question: Softmax Temperature

Softmax with temperature is:

$$
\hat{y}_i = \frac{e^{z_i/T}}{\sum_{j=1}^{K} e^{z_j/T}}
$$

where $T>0$.

1. What happens when $T=1$?
2. What happens when $T>1$?
3. What happens when $0<T<1$?
4. Draw three probability bar charts for the same logits with low, normal, and high temperature.
5. Explain why high temperature can increase diversity in generated text.
6. Explain why low temperature can make output more deterministic.
7. Give one situation where low temperature is preferred.
8. Give one situation where higher temperature is preferred.

## Question: Softmax in Attention

The attention formula contains:

$$
\mathrm{Softmax}\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}\right)
$$

1. In words, what does softmax do to attention scores?
2. Why should attention weights sum to 1 across the tokens being attended to?
