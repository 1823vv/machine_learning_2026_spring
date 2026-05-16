## Question: Train / Validation / Test Split

In model selection, data is often split into training, validation, and test sets. Each split has a different role.

1. What is the training set used for?
2. What is the validation set used for?
3. What is the test set used for?
4. Draw a workflow showing: train models, compare validation performance, choose a model, and evaluate once on the test set.
5. Why should the test set not be used repeatedly during model selection?
6. What is data leakage?
7. Give one example of data leakage in preprocessing or model selection.

## Question: K-Fold Cross-Validation

K-fold cross-validation estimates generalization performance by splitting the training data into $K$ folds and rotating which fold is used for validation.

1. Explain the steps of K-fold cross-validation.
2. If $K=5$, how many times is the model trained?
3. Why can cross-validation be useful when the dataset is small?
4. Why is cross-validation more expensive than a single train/validation split?
5. After cross-validation selects a hyperparameter, what data should be used to train the final model before the final test evaluation?
