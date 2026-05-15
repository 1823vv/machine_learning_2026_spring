## Question: Full-Batch GD versus Mini-Batch SGD

For a dataset with $n$ samples, full-batch gradient descent uses:

$$
g = \frac{1}{n}\sum_{i=1}^{n}\frac{\partial \mathcal{L}_i}{\partial W}
$$

Mini-batch SGD uses:

$$
g = \frac{1}{B}\sum_{i\in\mathcal{B}}\frac{\partial \mathcal{L}_i}{\partial W}
$$

1. Explain the meaning of $B$ and $\mathcal{B}$.
2. What is the batch size for full-batch GD?
3. What is the batch size for one-sample SGD?
4. What is the usual meaning of “SGD” in this course?
5. Draw two optimization paths:
   - a smooth full-batch GD path,
   - a noisy mini-batch SGD path.
6. Why is full-batch GD expensive for very large datasets?
7. Why can mini-batch SGD be faster in practice?
8. Why can mini-batch noise sometimes help escape saddle points or shallow bad regions?

## Question: Batch Size Trade-off

1. What happens when the batch size is very small?
2. What happens when the batch size is very large?
3. Why does batch size interact with learning rate?
