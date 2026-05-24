## Question: Self-Attention vs Cross-Attention

In encoder-decoder transformers (e.g. *Attention is All You Need*), both self-attention and cross-attention are used. Together they let a model represent the current target-side context while also reading useful information from source-side representations.

1. Write the common scaled dot-product attention formula.
2. In a standard Transformer self-attention layer, explicitly state where the input vectors for $Q$, $K$, and $V$ originate. 
3. In an encoder-decoder cross-attention layer, explicitly state where the input vectors for the queries $Q$ originate, and where the vectors for the keys $K$ and values $V$ originate.
4. Explain why cross-attention is crucial for traditional sequence-to-sequence tasks like machine translation. How does it allow the model to align target tokens with source tokens?
5. State whether state-of-the-art autoregressive large language models such as DeepSeek and ChatGPT utilize a decoder-only, encoder-only, or encoder-decoder architecture. Based on your answer, explain whether these models require cross-attention layers during text generation.