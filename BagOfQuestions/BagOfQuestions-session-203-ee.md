## Question: Causal Mask for Autoregressive Decoding

In next-token generation, each token should use only itself and earlier tokens, not future tokens. A causal mask enforces this rule inside self-attention.

1. For query position $i$ and key position $j$, state the basic causal validity rule using $j \le i$ or $j > i$.
2. For sequence length $n=4$, which key positions can query position $i=3$ attend to under causal masking?
3. In one sentence, explain why missing a causal mask can make training look better than real generation quality.
4. In implementation, masked positions are often set to $-\infty$ before softmax. Why does this effectively block those positions?
5. In one sentence, explain why causal masking is required even when full sequences are available during training.
