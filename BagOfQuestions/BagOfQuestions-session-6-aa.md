## Question: Generalization, Overfitting, and Data Splits

A machine learning model is useful only if it performs well on unseen data, rather than merely memorizing the training examples it has already encountered. Suppose a classifier achieves 99% accuracy on its training set but only 75% accuracy on a validation set sampled from the same data distribution.

1. Define the difference between training performance and generalization performance. Based on the scenario above, what does the 24% performance gap suggest about the model's current state?
2. Explain why a model that achieves a very low training loss or near-perfect training accuracy can still be a poor model. Why is memorizing training examples fundamentally different from learning a useful underlying pattern?
3. Sketch a single plot illustrating the behavior of an overfitting model over training epochs. Draw two distinct curves on this plot: one for **training loss** and one for **validation loss**.
4. On your drawing, clearly mark the exact point or region where the model begins to overfit. Use your plot to explain why a validation dataset is essential for detecting this boundary, and why the validation data must *never* be used to update the model’s parameters directly during gradient descent.
5. Provide two distinct regularization or data-driven strategies that can be implemented to narrow this generalization gap and improve performance on unseen data.
