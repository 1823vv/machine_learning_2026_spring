## Question: A Neural Network That Memorizes Too Well

You train a neural network for image classification. After many epochs, training accuracy is almost 100%, but validation accuracy is much lower.

1. What does this symptom suggest?
2. Draw training and validation accuracy curves for this situation. Draw also the training and validation loss curves for this situation.
3. Explain why increasing model capacity can make this problem worse on a small dataset.
4. Propose two regularization or data strategies that could improve validation performance.

## Question: Early Stopping with Validation Loss

Early stopping stops training when validation performance no longer improves, even if training loss continues decreasing.

1. Draw a validation-loss curve where early stopping would be useful. Mark the epoch with the best validation loss.
2. Explain the idea of “patience” in early stopping.
3. Why might stopping at the lowest training loss be a bad idea?
4. Explain how early stopping acts like a regularization method.
