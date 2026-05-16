## Question: Generalization Is the Real Goal

A machine-learning model is useful only if it performs well on unseen data from the same problem, not merely on the training examples it has already seen.

1. Explain the difference between training performance and generalization performance.
2. Why can a model with very low training loss still be a bad model?
3. Draw two curves over training epochs: training loss and validation loss for an overfitting model.
4. Mark on your drawing the region where validation loss starts getting worse while training loss continues improving.
5. Explain why validation data is useful for detecting this situation.
6. Give two possible strategies to improve generalization.

## Question: Seen Data versus Unseen Data

Suppose a classifier obtains 99% accuracy on the training set but only 75% accuracy on a validation set sampled from the same distribution.

1. What does this gap suggest?
2. Why is memorizing training examples not the same as learning a useful pattern?
3. What additional information would you want before deciding whether the model is acceptable?
4. Draw a simple schema showing training data, validation data, and future unseen data.
5. Explain why the validation set should not be used to update model parameters directly.
