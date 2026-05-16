## Question: Momentum Formula and Valley Intuition

Mini-batch SGD can zig-zag in narrow valleys of the loss surface. Momentum adds a velocity term $v$ that averages gradients over time:

$$
v \leftarrow \beta v + (1-\beta)g,
$$

$$
W \leftarrow W - \eta v.
$$

1. Write the momentum update formulas from memory.
2. Explain the meaning of $g$, $v$, $\beta$, and $\eta$.
3. What happens to $v$ when gradients point in a consistent direction for many steps?
4. What happens to $v$ when gradients alternate directions because of oscillation?
5. Draw a narrow valley and compare the path of plain mini-batch SGD with the path of momentum.
6. Explain why momentum can reduce zig-zagging.
7. Explain why momentum can accelerate movement along a shallow but consistent direction.
8. What is a common value for $\beta$?

## Question: Momentum as a Physical Analogy

Momentum is often compared to a ball rolling downhill. This analogy can help intuition, but it is not a perfect description of the algorithm.

1. In the analogy, what does “velocity” represent?
2. In the analogy, what does “inertia” represent?
3. How does the analogy explain faster movement along a consistent downhill direction?
4. How does the analogy explain reduced oscillation across a narrow valley?
5. Give one way in which the physical analogy can be misleading.
