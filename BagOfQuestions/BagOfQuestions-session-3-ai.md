## Question: One-Hot Encoding for Categorical Input Features

A machine learning dataset contains three continuous numerical input features and one categorical input feature called `State`. The possible values for `State` are `New York`, `California`, and `Florida`. Because mathematical models require purely numeric input vectors, this categorical feature must be transformed using one-hot encoding.

1. Construct a reference table showing the exact one-hot encoded vector representation for each of the three distinct state values.
2. Given the 3 original numerical features, what is the total input dimension ($d_{\mathrm{in}}$) of the final processed feature matrix? For an input sample belonging to `California`, write out the full, combined input row vector $x$ using placeholders (e.g., $x_1, x_2, x_3$) for the numerical features followed by the explicit numeric values of the encoded state.
3. Explain why mapping these categorical states to single integer labels (such as `New York = 0`, `California = 1`, and `Florida = 2`) introduces an inappropriate inductive bias for a linear model.
