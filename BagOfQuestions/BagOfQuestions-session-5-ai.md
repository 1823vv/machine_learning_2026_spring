## Question: Mini-Batch Training Logs and Generalization

During training, you see these logs:

```text
Epoch 1:  train accuracy 45%, validation accuracy 43%
Epoch 5:  train accuracy 82%, validation accuracy 78%
Epoch 20: train accuracy 99%, validation accuracy 81%
```

1. What does the improvement from epoch 1 to epoch 5 suggest?
2. What does the large gap by epoch 20 suggest?
3. Why should both training and validation accuracy be monitored?
4. Draw training and validation accuracy curves that match this story.
5. How can mini-batch noise affect training curves compared with full-batch GD?
6. Which later Session 7 techniques might help if validation accuracy stops improving?
7. Why should the final test set not be used to decide the optimizer settings?

## Question: Loss Curves for Different Optimizers

Draw possible validation-loss curves for:

1. learning rate too large,
2. learning rate too small,
3. Adam with a reasonable learning rate,
4. SGD with noisy but improving progress.
