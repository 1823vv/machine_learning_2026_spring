## Question: L1 and L2 Optimization View

Regularized training can be viewed as minimizing a loss while penalizing the size of weights. L1 and L2 penalties have different mathematical shapes.

1. Write the L1 penalty using double pipes for absolute value.
2. Write the L2 penalty. Draw the contour shape of an L1 constraint in two dimensions.
3. Draw the contour shape of an L2 constraint in two dimensions. Explain why L1 is more likely to produce exactly zero weights.
4. Explain why L2 usually shrinks weights smoothly instead of setting many exactly to zero. In one sentence, connect regularization to model generalization.

## Question: Smooth versus Non-Smooth

The L2 penalty is smooth, while the L1 penalty has a sharp corner at zero. This difference matters for optimization and sparsity.

1. Draw the one-dimensional functions $\|w\|$ and $w^2$.
2. Where is the L1 penalty non-smooth? Why does the L1 corner help produce sparse solutions?
3. Why is L2 often easier to optimize with gradient-based methods?
4. Give one practical reason to prefer sparse models.
