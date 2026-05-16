## Question: Full-Batch Gradient Descent in a NumPy Neural Network

In this NumPy neural-network training loop, the entire training dataset is used for each update.

1. Explain what “full-batch gradient descent” means, and identify where the whole `X_train` and `y_train` are passed into `train` at once.
2. Give one advantage and one disadvantage of full-batch gradient descent, especially for large datasets.
3. Explain how this idea changes when using mini-batches or mini-batch SGD.
4. Draw a comparison between:

   ```text
   full-batch GD: all samples -> one update
   mini-batch SGD: small batch -> one update, repeated many times
   ```

## Question: Loss Trend

During training, loss and accuracy are monitored to catch optimization or implementation problems.

1. Should the loss decrease every single epoch perfectly? Explain.
2. What would you check if the loss becomes `nan`?
3. What would you check if accuracy stays around 10% on MNIST?
