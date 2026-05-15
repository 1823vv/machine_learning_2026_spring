## Question: L1 and L2 Regularization — Formula, Geometry, and Intuition

For a model with weight matrices $W^{(1)}, W^{(2)}, \dots, W^{(L)}$, we add regularization to reduce overfitting.

1. Write a possible L1 penalty formula over all weight matrices.
2. Write a possible L2 penalty formula over all weight matrices.
3. Write a total loss formula of the form:

   $$
   \mathcal{L}_{total}=\mathcal{L}_{data}+\text{regularization penalty}.
   $$

4. Which regularization method is more associated with sparsity and feature selection: L1 or L2?
5. Explain why L1 can push some weights exactly to zero more naturally than L2.
6. Draw the classic geometry picture for L1 and L2 regularization:
   - L1 constraint as a diamond,
   - L2 constraint as a circle,
   - loss contours as ellipses,
   - the first touching point.
7. Use your drawing to explain why L1 often produces sparse solutions.

## Question: Weight Penalty and Neural Network Capacity

1. Explain why very large weights can make a neural network more sensitive to small input changes.
2. Explain why penalizing weights can improve generalization.
3. Compare L2 regularization and dropout: one penalizes parameters directly, the other injects noise into hidden activations. How are both trying to fight overfitting?
