## Question: Design a Small Neural Network for Sentiment Analysis

In the Session 3 practice, students build a project for sentiment analysis using a neural network.

Suppose each movie review is converted into a vector of 1000 TF-IDF features.

Design a binary sentiment classifier.

1. Draw an architecture from the 1000-dimensional input vector to a positive/negative output.
2. Choose two hidden-layer widths and justify them briefly.
3. What should the final output dimension be for binary sentiment classification?
4. What final activation function should be used?
5. What loss function should be used?
6. Count the total number of trainable parameters for your proposed architecture.
7. Explain why a validation set is needed for this project.
8. Draw the full project flow:

   ```text
   raw reviews -> vectorization -> neural network -> probability -> class label -> deployed demo
   ```

## Question: Alternative Multiclass Version

Now suppose the task has 5 sentiment classes from “very negative” to “very positive.”

1. What should the final output dimension be?
2. What output activation should be used?
3. What loss function should be used?
4. How does parameter counting change in the final layer?
