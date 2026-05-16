## Question: One-Hot Encoding for Categorical Input Features

A regression dataset has three numerical input features and one categorical input feature called `State`. The possible states are `New York`, `California`, and `Florida`. A model needs numerical input vectors, so the categorical feature is one-hot encoded.

1. Write a one-hot encoding table for the three state values.
2. If the original input has 3 numerical features, what is the input dimension after using all 3 one-hot state columns? For one sample from `California`, write the state part of the one-hot vector.
3. Explain why using integer labels such as `0`, `1`, and `2` for the states can be misleading. In a linear model with a bias term, why might one of the three one-hot columns be dropped?
4. Draw a schema showing numerical features and one-hot state features concatenated into one input row vector $x$.
