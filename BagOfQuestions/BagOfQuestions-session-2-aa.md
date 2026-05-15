# Bag of Questions — Session 2 — aa

## Question: From Linear Regression to Logistic Regression

We want to build a binary classifier for a dataset with two input features, $x_1$ and $x_2$, and label $y \in \{0,1\}$.

1. Write the linear score $z$ of logistic regression using the row-vector convention from the course.
2. Write the sigmoid function $\sigma(z)$.
3. Write the full logistic regression model $\hat{y}$.
4. Explain why the output of ordinary linear regression is not suitable as a probability for binary classification.
5. Draw a schema showing the flow:

   ```text
   input features -> linear score -> sigmoid -> probability -> class decision
   ```

6. On your drawing, mark clearly which part is linear and which part is nonlinear.
7. Suppose $z=-2$, $z=0$, and $z=2$. Without using a calculator, rank the three corresponding predicted probabilities from smallest to largest, and explain why.

## Question: Short Code Reading

Consider this line from a logistic regression implementation:

```python
linear_model = np.dot(X, self.weights) + self.bias
```

1. If $X \in \mathbb{R}^{n \times d}$ and `self.weights` has shape $(d,)$, what is the shape of `linear_model`?
2. What mathematical expression does this line implement?
3. Why do we apply a sigmoid function after this line in logistic regression, but not in linear regression?
