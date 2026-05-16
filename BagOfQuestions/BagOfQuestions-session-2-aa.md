## Question: Logistic Regression Boundary and Losses

Consider binary logistic regression with two features. Under the row-vector convention, one example has $x \in \mathbb{R}^{1 \times 2}$, weights $W \in \mathbb{R}^{2 \times 1}$, bias $b \in \mathbb{R}^{1 \times 1}$, logit $z=xW+b$, and probability $\hat y=\sigma(z)$.

1. Write the equation of the decision boundary when the classification threshold is 0.5, and state what is special about this boundary in two dimensions.
2. Write the binary cross-entropy loss for one example and for $n$ examples.
3. Write the mean squared error loss that would be obtained if MSE were used with logistic regression probabilities.
4. Explain why binary cross-entropy is usually preferred over MSE for logistic regression.

## Question: GitHub CI/CD/Actions/Release Project

In our GitHub CI/CD/Actions/Release project, a trained model file such as `linear_model.txt` is generated automatically and then handled by the workflow.

1. Explain why SSH keys can be useful instead of HTTPS for automated workflow access.
2. Explain how `linear_model.txt` is generated, using one simple diagram.
3. Explain how GitHub Actions streamlines the development and release process.
