## Question: Full-Batch Gradient Descent in a NumPy Neural Network

In this NumPy neural-network training loop, the entire training dataset is used for each update.

1. What does “full-batch gradient descent” mean?
2. In the code, where do you see that the whole `X_train` and `y_train` are passed into `train` at once?
3. What is one advantage of full-batch gradient descent?
4. What is one disadvantage of full-batch gradient descent for large datasets?
5. How will this idea change later when using mini-batches or SGD?
6. Draw a comparison between:

   ```text
   full-batch GD: all samples -> one update
   mini-batch SGD: small batch -> one update, repeated many times
   ```

## Question: Loss Trend

1. Should the loss decrease every single epoch perfectly? Explain.
2. What would you check if the loss becomes `nan`?
3. What would you check if accuracy stays around 10% on MNIST?
