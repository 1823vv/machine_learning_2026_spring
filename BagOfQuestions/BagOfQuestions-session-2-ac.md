## Question: Decision Boundary

A logistic regression classifier with two features uses

$$
\hat{y}=\sigma(w_1x_1+w_2x_2+b)
$$

and predicts class 1 when $\hat{y}>0.5$.

1. Explain why $\hat{y}>0.5$ is equivalent to $w_1x_1+w_2x_2+b>0$.
2. Write the decision boundary equation, then rewrite it as $x_2=ax_1+c$ assuming $w_2 \neq 0$.
3. Explain what is special about the decision boundary of logistic regression with two input features.
4. Does applying the nonlinear sigmoid make the decision boundary nonlinear? Explain.

## Question: Numerical Boundary

For the model

$$
\hat{y}=\sigma(2x_1-x_2-1),
$$

use threshold 0.5.

1. Write the decision boundary equation and convert it into the form $x_2=ax_1+c$.
2. Classify the point $(x_1,x_2)=(2,1)$.
3. Classify the point $(x_1,x_2)=(0,0)$.
