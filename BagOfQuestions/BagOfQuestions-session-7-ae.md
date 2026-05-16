## Question: Data Augmentation as Label-Preserving Transformation

Data augmentation creates new training examples by applying transformations that should not change the label. For example, a small shift of an MNIST digit usually keeps the same digit label.

1. Define data augmentation and explain how it modifies the effective capacity and generalization behavior of a machine learning model.
2. Provide three distinct examples of label-preserving transformations commonly used for computer vision or image classification tasks.
3. Describe the primary challenges encountered when designing data augmentation strategies for natural language processing (text data) compared to computer vision (image data). Give one example of a text-based transformation.



## Question: L1 and L2 Regularization — Formula, Geometry, and Intuition

In linear regression, regularization helps prevent overfitting by adding penalty terms to the loss function. Let $\mathcal{L}_{data}(w)$ be the original data loss and $\lambda > 0$ be the regularization strength.

1. Write the complete objective function for linear regression with L1 regularization (Lasso).
2. Write the complete objective function for linear regression with L2 regularization (Ridge).
3. Which regularization method (L1 or L2) can produce exactly zero weights? Which one tends to shrink weights smoothly?
4. In 2D space with weights $w_1$ and $w_2$, draw two figures:
   - First figure: Show data loss contours (ellipses) and the L1 constraint region. Mark where a contour first touches the constraint region.
   - Second figure: Show data loss contours (ellipses) and the L2 constraint region. Mark where a contour first touches the constraint region.
5. Based on your drawings, explain why one method leads to sparse solutions (feature selection) while the other does not.
6. Why is $\lambda$ considered a hyperparameter that needs tuning?