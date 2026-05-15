# Bag of Questions — Session 2 — ac

## Question: Decision Boundary Story

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
6. Draw a 2-D feature space with:
   - $x_1$ on the horizontal axis,
   - $x_2$ on the vertical axis,
   - a straight decision boundary,
   - one side labeled class 0 and the other side labeled class 1,
   - a few sample points on both sides.
7. Add arrows or color shading to your drawing to show where $z>0$, $z=0$, and $z<0$.

## Question: Numerical Boundary

For the model:

$$
\hat{y}=\sigma(2x_1-x_2-1)
$$

1. Write the decision boundary equation.
2. Convert it into the form $x_2 = ax_1+c$.
3. Classify the point $(x_1,x_2)=(2,1)$ using the threshold $0.5$.
4. Classify the point $(x_1,x_2)=(0,0)$ using the threshold $0.5$.
