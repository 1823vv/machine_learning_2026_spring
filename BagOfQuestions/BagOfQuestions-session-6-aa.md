## Question: Generalization Is the Real Goal

A machine-learning model is useful only if it performs well on unseen data from the same problem, not merely on the training examples it has already seen.

1. Explain the difference between training performance and generalization performance.
2. Why can a model with very low training loss still be a bad model? Draw two curves over training epochs: training loss and validation loss for an overfitting model.
3. Mark on your drawing the region where validation loss starts getting worse while training loss continues improving. Explain why validation data is useful for detecting this situation.
4. Give two possible strategies to improve generalization.

## Question: Seen Data versus Unseen Data

Suppose a classifier obtains 99% accuracy on the training set but only 75% accuracy on a validation set sampled from the same distribution.

1. What does this gap suggest?
2. Why is memorizing training examples not the same as learning a useful pattern? What additional information would you want before deciding whether the model is acceptable?
3. Explain why the validation set should not be used to update model parameters directly.
