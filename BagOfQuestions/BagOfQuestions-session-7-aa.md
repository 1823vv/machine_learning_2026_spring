## Question: A Neural Network That Memorizes Too Well

A neural network for image classification has very low training loss, but its validation loss starts increasing after some epochs.

1. Explain what overfitting means in this story.
2. Draw a training-loss and validation-loss curve as a function of epoch. Your drawing should show the early stage, the best validation epoch, and the overfitting region.
3. Mark the epoch

   $$
   t^* = \arg\min_t \mathcal{L}_{val}(t)
   $$

   on your drawing.
4. Explain why choosing the final epoch is not always the best model-selection decision.
5. Explain why early stopping can be viewed as a regularization method, even though it does not add an explicit penalty term to the loss.
6. In one sentence, compare early stopping with L2 regularization: what does each one control?
7. If patience is $k=5$, describe in words when training should stop.

## Question: From Session 1 to Session 7

In Session 1, you learned linear regression. In Session 7, you use validation loss to stop training.

1. Why is validation loss more relevant than training loss for selecting the final model?
2. Draw three fitted curves on the same regression dataset: underfitting, good fitting, and overfitting.
3. For each curve, state whether bias is high or low and whether variance is high or low.
