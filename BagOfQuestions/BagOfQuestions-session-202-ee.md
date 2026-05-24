## Question: Sinusoidal Positional Encoding and the Attention Pipeline

In the transformer architecture, self-attention is permutation-invariant with respect to token positions: it computes the same similarity scores regardless of where tokens appear in the sequence. To preserve order information, a positional encoding vector is added to each token embedding before the first attention layer.

Let $d_{\text{model}}$ denote the embedding dimension. For a token at position $pos$ (0-indexed), the standard sinusoidal positional encoding $PE_{pos} \in \mathbb{R}^{d_{\text{model}}}$ is defined as follows, where $i$ indexes the sine/cosine pairs:

$$
\boxed{PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\,2i/d_{\text{model}}}}\right)}
$$

$$
\boxed{PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\,2i/d_{\text{model}}}}\right)}
$$

After computing the positional encodings, each token embedding $x_i$ is combined with its positional vector $p_i = PE_i$ to form the input to the attention layers.

1. For position $pos = 0$, compute the values in the sine channels (even indices) and the cosine channels (odd indices). Explain why the result is the same regardless of $d_{\text{model}}$.

2. In one sentence, explain why using many different frequencies (different values of $i$) is helpful for representing position.

3. Why does this sinusoidal method not require learning extra positional parameters?

4. Write the formula that combines token embedding $x_i$ with its positional vector $p_i$ to produce the representation fed into attention.

5. If two tokens are the same word but appear at different positions, why can their combined vectors still differ?

6. After adding positional information, the next high-level step is scaled dot-product attention. Explain in simple words why injecting position information *before* attention helps the model with next-token prediction.

