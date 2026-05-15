## Question: L1 and L2 Optimization View

For L2 regularization, the objective includes:

$$
\lambda \sum_{j=1}^{d} W_j^2
$$

For L1 regularization, the objective includes:

$$
\lambda \sum_{j=1}^{d} \|W_j\|
$$

1. Write the gradient contribution of L2 regularization for weight $W_j$.
2. Write the gradient contribution of L1 regularization for weight $W_j$ away from zero.
3. Write the L2 update rule for $W_j$ using leftarrow notation.
4. Rewrite the L2 update to show the multiplicative shrinkage term on $W_j$.
5. Explain why L2 is called weight decay.
6. Write the L1 update rule for $W_j$ using leftarrow notation.
7. Explain why L1 can drive weights exactly to zero more naturally than L2.
8. Draw a number line showing how a positive weight and a negative weight are both pushed toward zero by L1.

## Question: Smooth versus Non-Smooth

1. Why is L2 smooth at zero?
2. Why is L1 non-smooth at zero?
3. How does this difference relate to sparsity?
