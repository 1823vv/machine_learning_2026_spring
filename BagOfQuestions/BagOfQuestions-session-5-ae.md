## Question: Momentum Formula and Valley Intuition

Mini-batch SGD can zig-zag in narrow valleys. Momentum adds a velocity term:

$$
v \leftarrow \beta v + (1-\beta)g
$$

$$
W \leftarrow W - \eta v
$$

1. Write the momentum update formulas from memory.
2. Explain the meaning of $g$, $v$, $\beta$, and $\eta$.
3. What happens when gradients point in a consistent direction for many steps?
4. What happens when gradients alternate directions because of oscillation?
5. Draw a narrow valley and compare the SGD path with the momentum path.
6. Explain why momentum can reduce zig-zagging.
7. Explain why momentum can accelerate movement along a shallow but consistent direction.
8. What is a common value for $\beta$?

## Question: Physical Analogy

Momentum is often compared to a ball rolling downhill.

1. What does “velocity” mean in this analogy?
2. What does “inertia” mean in this analogy?
3. Where can this analogy be misleading?
