## Question: Momentum

Mini-batch SGD can zig-zag in narrow valleys of the loss surface. Momentum adds a velocity term $v$ that averages gradients over time.
1. Write the momentum update formulas from memory. Explain the meaning of $g$, $v$, $\beta$, and $\eta$.
2. What happens to $v$ when gradients point in a consistent direction for many steps? What happens to $v$ when gradients alternate directions because of oscillation?
3. Explain why momentum can accelerate movement along a shallow but consistent direction. What is a common value for $\beta$?
