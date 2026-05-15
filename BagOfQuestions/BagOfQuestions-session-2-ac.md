## Question: Decision Boundary

A logistic regression classifier with two features uses:

$$
\hat{y}=\sigma(w_1x_1+w_2x_2+b)
$$

and predicts class 1 when $\hat{y}>0.5$.

1. Explain why the condition $\hat{y}>0.5$ is equivalent to $w_1x_1+w_2x_2+b>0$.
2. Write the equation of the decision boundary.
3. Rewrite the boundary in the form $x_2 = ax_1 + c$, assuming $w_2 \neq 0$.
4. What is special about the decision boundary of logistic regression with two input features?
5. Does applying the nonlinear sigmoid make the decision boundary nonlinear? Explain.

## Question: Numerical Boundary

For the model:

$$
\hat{y}=\sigma(2x_1-x_2-1)
$$

1. Write the decision boundary equation.
2. Convert it into the form $x_2 = ax_1+c$.
3. Classify the point $(x_1,x_2)=(2,1)$ using the threshold $0.5$.
4. Classify the point $(x_1,x_2)=(0,0)$ using the threshold $0.5$.
