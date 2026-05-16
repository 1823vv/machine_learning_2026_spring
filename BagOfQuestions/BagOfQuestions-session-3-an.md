## Question: ChatGPT-Style Tiny Language Model Parameter Counting

A company uses a tiny ChatGPT-style demo to show how a language model turns words into numbers and then predicts the next word. This is not the full real architecture; it is a simplified counting exercise. The vocabulary has $V=1000$ possible tokens. Each token is represented by a vector of length $d=10$.

The toy model has three trainable parts: an embedding table $E \in \mathbb{R}^{1000 \times 10}$, one hidden dense layer with weight matrix $W_h \in \mathbb{R}^{10 \times 20}$ and bias vector $b_h \in \mathbb{R}^{1 \times 20}$, and an output vocabulary layer with weight matrix $W_{out} \in \mathbb{R}^{20 \times 1000}$ and bias vector $b_{out} \in \mathbb{R}^{1 \times 1000}$.

1. Count the trainable parameters in the embedding table. Explain in one sentence what this table stores.
2. Count the weights and biases in the hidden dense layer.
3. Count the weights and biases in the output vocabulary layer.
4. Add the three counts to get the total number of trainable parameters in this toy model.
5. Explain why the output layer is large even though the hidden layer is small.
