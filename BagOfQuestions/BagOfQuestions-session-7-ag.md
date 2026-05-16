## Question: Grid Search versus Random Search

You need to tune two hyperparameters: learning rate $\eta$ and regularization strength $\lambda$. Candidate values are

$$
\eta \in \{0.1, 0.01, 0.001\},
$$

$$
\lambda \in \{0, 0.001, 0.01, 0.1\}.
$$

1. How many configurations does grid search evaluate?
2. Draw the grid-search points in a 2-D hyperparameter plane.
3. Explain why grid search becomes expensive when the number of hyperparameters grows.
4. Explain the core idea of random search.
5. Draw random-search points in the same 2-D plane.
6. Suppose $\eta$ matters much more than $\lambda$. Explain why random search may be more efficient than grid search under the same trial budget.
7. In one sentence, explain why hyperparameter optimization is often called black-box optimization.

## Question: Trial Budget

A practitioner only has enough compute budget to train 20 candidate models while tuning hyperparameters.

1. Why is training one model considered expensive?
2. How would you decide which hyperparameter ranges to search first?
3. Why should the test set not be used repeatedly during hyperparameter tuning?
4. Why is it useful to record every trial's hyperparameters and validation score?
5. How might you refine the search after seeing the first few trials?
