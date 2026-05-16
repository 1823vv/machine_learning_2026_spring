## Question: ChatGPT Vocabulary Table and Shared Weights

In a simplified ChatGPT-like next-token predictor, the vocabulary has $V=1000$ possible tokens and each token vector has length $d=10$. The model has an embedding table $E \in \mathbb{R}^{1000 \times 10}$ that turns token IDs into vectors.

At the end, the model needs one logit for each possible next token. One design uses a separate output matrix $W_{out} \in \mathbb{R}^{10 \times 1000}$ and an output bias $b_{out} \in \mathbb{R}^{1 \times 1000}$. Another design reuses the embedding table as the output matrix, so there is no separate $W_{out}$, but the output bias is still trainable.

1. Count the parameters in the embedding table $E$.
2. Count the parameters in the separate output matrix $W_{out}$ and output bias $b_{out}$.
3. If the model reuses the embedding table and keeps only the output bias as extra output parameters, how many output-side parameters remain?
4. How many parameters are saved by reusing the embedding table instead of storing a separate $W_{out}$?
5. Explain why the model needs $1000$ output logits when the vocabulary has $1000$ tokens.
