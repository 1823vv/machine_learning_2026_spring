## Question: Input One-Hot versus Output One-Hot

One-hot encoding can appear in two different places: input features and output labels. These two uses look similar but have different meanings.

1. Give one example of one-hot encoding used for an input feature.
2. Give one example of one-hot encoding used for an output label.
3. For input one-hot encoding, explain how the input dimension $d$ changes.
4. For output one-hot labels, explain how the number of output logits $K$ is determined.
5. Which weight matrix shape is affected by input one-hot features: $W^{(1)}$ or $W^{(L)}$? Explain.
6. Which weight matrix shape is affected by the number of output classes: $W^{(1)}$ or $W^{(L)}$? Explain.
7. In softmax cross-entropy, why does the one-hot output label help produce the simple gradient signal $\hat{Y}-Y$?
8. Draw a two-column comparison of input one-hot encoding and output one-hot labels.
