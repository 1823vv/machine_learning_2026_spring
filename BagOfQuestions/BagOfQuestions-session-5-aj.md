## Question: Debugging Neural Network Optimization Without Writing New Optimizer Code

A neural network on MNIST is not learning. Accuracy stays near 10%.

You are not asked to write optimizer code. Instead, design a debugging plan.

1. Why is 10% accuracy suspicious for MNIST?
2. What data shapes would you print first?
3. What activation or logits shapes would you print through the network?
4. Why should you check whether gradients are nonzero?
5. Why should you check whether parameters actually change after an optimizer step?
6. Why should you check for `nan` or extremely large loss values?
7. Draw a debugging flowchart:

   ```text
   data shapes -> forward outputs -> loss -> gradients -> parameter update -> accuracy
   ```

8. Explain how a too-large learning rate could create `nan` values.
9. Explain how a too-small learning rate could make learning look stuck.

## Question: Assertions and Breakpoints

1. What is one useful assertion for a Dense layer input shape?
2. What is one useful assertion for logits shape in MNIST?
3. When is a breakpoint more useful than print debugging?
4. Why should breakpoints be removed before committing code?
