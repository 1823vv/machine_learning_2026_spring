# Bag of Questions — Session 2 — ai

## Question: Probability, Threshold, and Business Meaning

A logistic regression model predicts whether a user will buy a product. The output is:

$$
\hat{y}=P(y=1\mid x)
$$

where class 1 means “will buy”.

1. If $\hat{y}=0.8$, explain the meaning in one sentence.
2. With threshold 0.5, what class is predicted?
3. With threshold 0.9, what class is predicted?
4. Explain why changing the threshold changes the decision rule but does not change the trained logistic regression probability model.
5. Draw a number line from 0 to 1 showing:
   - probabilities,
   - threshold 0.5,
   - threshold 0.9,
   - regions predicted as class 0 and class 1 for each threshold.
6. In an application where false positives are very expensive, would you prefer a higher or lower threshold for predicting class 1? Explain briefly.

## Question: Link to the Decision Boundary

For two features, the model is:

$$
\hat{y}=\sigma(w_1x_1+w_2x_2+b)
$$

1. If the threshold is 0.5, what equation defines the decision boundary?
2. If the threshold is changed to $t$, where $0<t<1$, the boundary satisfies $\sigma(z)=t$. Explain whether the boundary is still a line in the $(x_1,x_2)$ plane.
3. For threshold $t=0.5$, why is the boundary especially simple?
