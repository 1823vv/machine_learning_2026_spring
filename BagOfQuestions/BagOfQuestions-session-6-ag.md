## Question: L1 and L2 Regularization Formulas

Regularization modifies the objective:

$$
\min_W \mathcal{L}_{train}(W)+\lambda\Omega(W)
$$

1. Explain the role of $\mathcal{L}_{train}(W)$.
2. Explain the role of $\Omega(W)$.
3. Explain the role of $\lambda$.
4. Write the L2-regularized objective.
5. Write the L1-regularized objective, using the course convention for absolute value.
6. Which one is also called weight decay?
7. Which one is more associated with sparsity and feature selection?
8. Create a comparison table for L1 and L2 with rows:
   - penalty shape,
   - smoothness,
   - sparsity,
   - effect on weights.

## Question: Large Weights

1. Why can large weights make predictions sensitive to small input noise?
2. Why can large weights be connected to overfitting?
3. Explain how regularization changes the goal from pure data fitting to data fitting plus complexity control.
