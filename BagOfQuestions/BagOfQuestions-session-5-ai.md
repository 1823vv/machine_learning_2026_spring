## Question: Mini-Batch Training Logs and Generalization

During neural-network training, suppose the training loss decreases steadily, training accuracy increases, but validation accuracy stops improving after some epoch.

1. Draw possible training-loss and validation-loss curves for this situation.
2. Explain why training performance can keep improving while validation performance stops improving. What does this suggest about generalization?
3. Name two techniques that might help when validation accuracy stops improving.
4. Why should the final test set not be used to decide when to stop training?

## Question: Loss Curves for Different Optimizers

Suppose the same neural network is trained with plain mini-batch SGD, momentum, and Adam using reasonable learning rates. Their loss curves may look different even on the same dataset.

1. Draw a possible training-loss curve for plain mini-batch SGD.
2. Draw a possible training-loss curve for SGD with momentum. Draw a possible training-loss curve for Adam.
3. Explain why mini-batch loss curves can be noisy.
4. Explain why a smoother or faster training-loss decrease does not automatically prove better test performance.
