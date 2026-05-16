## Question: Feature Scaling and Gradient Descent Geometry

A binary classifier uses two features:

- age, usually between 0 and 100,
- yearly income, possibly between 0 and 1000000.

The logistic regression gradient for the weights can be written as

$$
\frac{\partial \mathcal{L}}{\partial W}=\frac{1}{n}X^T(\hat{Y}-Y).
$$

1. Explain why very different feature scales can create problems for gradient descent, using the gradient formula above.
2. Draw loss contours before feature scaling, using elongated ellipses and a zig-zag gradient descent path.
3. Draw loss contours after good feature scaling, using more circular contours and a more direct path to the minimum.
4. Explain why feature scaling can make training faster even if it does not change the meaning of the classification task.
