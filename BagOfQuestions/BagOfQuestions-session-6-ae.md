## Question: Train / Validation / Test Split

A dataset is split into train, validation, and test sets.

1. What is the training set used for?
2. What is the validation set used for?
3. What is the test set used for?
4. Write two typical split ratios.
5. Why should the test set be used only once at the end?
6. Explain information leakage in the context of repeated test-set use.
7. Draw a workflow:

   ```text
   train model -> evaluate validation -> adjust hyperparameters -> final test once
   ```

8. Explain why using validation accuracy to choose a model is better than using training accuracy.

## Question: K-Fold Cross-Validation

1. Explain K-fold cross-validation in words.
2. For $K=5$, how many times do we train the model?
3. Why can K-fold cross-validation reduce variance in performance estimates?
4. What is the computational cost of K-fold cross-validation?
5. When might a single train/validation/test split be sufficient?
