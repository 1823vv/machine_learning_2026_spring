## Question: Causal Mask for Autoregressive Decoding

In next-token generation, each token should use only itself and earlier tokens, not future tokens. A causal mask enforces this rule inside self-attention.

1. Let $i$ denote the query index (representing the position generating the query) and $j$ denote the key index (representing the position of the token being attended to) in a sequence. State the basic mathematical relationship between $i$ and $j$ that determines whether an attention connection is causally valid.
2. Suppose a sequence has a length of $n = 4$, using 0-based indexing ($0, 1, 2, 3$). List the exact key positions $j$ that query position $i = 2$ is permitted to attend to under strict causal masking rules.
3. Explain why omitting a causal mask during training leads to target data leakage. Why does this leakage cause the model's training loss to appear exceptionally low while resulting in highly corrupted or broken generation quality during actual inference?
4. In implementation, masked positions are often set to $-\infty$ before softmax. Why does this effectively block those positions?
5. In one sentence, explain why causal masking is required even when full sequences are available during training.

