## Question: Feature Scaling and Gradient Descent Geometry

A binary classifier uses two features:

- age, usually between 0 and 100,
- yearly income, possibly between 0 and 1000000.

1. Explain why these very different feature scales can create problems for gradient descent.
2. Use the logistic regression gradient idea

   $$
   \frac{\partial \mathcal{L}}{\partial W}=\frac{1}{n}X^{\mathsf{T}}(\hat{Y}-Y)
   $$

   to explain why larger feature values can create larger weight updates.
3. Draw the loss contours before feature scaling. Use elongated ellipses and show a zig-zag gradient descent path.
4. Draw the loss contours after good feature scaling. Use more circular contours and show a more direct path to the minimum.
5. Explain why feature scaling can make training faster, even if it does not change the meaning of the classification task.

