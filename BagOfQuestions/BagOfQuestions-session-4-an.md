## Question: Data Loading and Shapes for MNIST

The data-loading function reads MNIST CSV files and returns:

```python
X_train, y_train, X_val, y_val, X_test, y_test
```

1. Why are pixel values divided by `255.0`?
2. What is the expected number of columns in `X_train` for MNIST images?
3. What does each row of `X_train` represent?
4. What does each value of `y_train` represent?
5. Why do we create a validation set in addition to a test set?
6. Why does the code set `np.random.seed(42)` before choosing validation indices?
7. Draw a diagram showing the split from full training data into training and validation subsets.

## Question: Validation and Test Sets

1. During model development, which set should be used to monitor model choices?
2. When should the test set be used?
3. Why is it a bad idea to repeatedly tune the model based on test accuracy?
