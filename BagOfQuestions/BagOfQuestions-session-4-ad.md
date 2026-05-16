## Question: Categorical Cross-Entropy Formula

For a multi-class classification problem with $n$ samples and $K$ classes, let $Y_{i,k}$ be the one-hot label and $\hat{Y}_{i,k}$ be the predicted probability.

1. Write the categorical cross-entropy loss over the full batch.
2. For one sample, if the true class is class 3, simplify the cross-entropy loss. Explain why a confident correct prediction has small loss.
3. Explain why a confident wrong prediction has large loss. Draw the curve of $-\log(p)$ as $p$ goes from 0 to 1.
4. On your drawing, mark where the model is confident and correct. On your drawing, mark where the model is confident and wrong.

## Question: One-Hot Labels

1. If there are 10 classes and the label is `7`, write the one-hot vector.
2. In the code, what is the shape of `one_hot_labels` if `logits.shape == (batch_size, 10)`?
3. Explain this line in words:

   ```python
   one_hot_labels[np.arange(batch_size), labels] = 1
   ```
