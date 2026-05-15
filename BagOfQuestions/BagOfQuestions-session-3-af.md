## Question: Softmax Shift Invariance and Numerical Stability

Softmax has the important property:

$$
\mathrm{softmax}(z)=\mathrm{softmax}(z+c)
$$

where the same constant $c$ is added to every logit.

1. Prove this property using the softmax formula.
2. Explain why $[2,1,0]$ and $[102,101,100]$ give the same softmax probabilities.
3. Explain the numerical danger of applying exponential directly to logits like $[1000,999,998]$.
4. Show how subtracting the maximum transforms $[1000,999,998]$ into a safer vector.
5. Write the stable softmax formula using $z_i-\max(z)$.
6. Draw a schema showing:

   ```text
   raw logits -> subtract max -> exponentiate -> normalize -> probabilities
   ```

## Question: Relative Gaps

1. Which distribution should be sharper: softmax of `[10, 2, -3]` or softmax of `[2, 1, 0]`?
2. Explain your answer using the idea of score gaps.
