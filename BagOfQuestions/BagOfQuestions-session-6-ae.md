## Question: Diagnosing the Bias–Variance Tradeoff and Learning Curves

The bias–variance tradeoff provides a fundamental framework for understanding model complexity, underfitting, and overfitting. We can analyze a model's behavior by looking at its performance relative to both **model complexity** and **training dataset size**.

### Part A: Error vs. Model Complexity

1. Define the terms **high bias** and **high variance** in the context of model performance on training and validation datasets. Provide one example of a model architecture (or configuration) that typically exhibits high bias, and one that typically exhibits high variance.
2. Sketch a single plot illustrating how error changes as **Model Complexity** increases (from low to high along the x-axis). Draw two distinct curves on this plot: one for **training error** and one for **validation error**.
3. On your sketch from sub-question 2, clearly partition and label three distinct zones: the **underfitting region**, the **overfitting region**, and the **optimal model-complexity region**.

### Part B: Diagnosing via Learning Curves

4. Sketch a new plot showing the **Learning Curves** for a **high-variance model**, where the x-axis represents the **Training Set Size** (from small to large) and the y-axis represents error. Draw both the training error and validation error curves. Use this sketch to explain why collecting more training data is an effective strategy for reducing variance.
5. Sketch a final plot showing the **Learning Curves** for a **high-bias model** over an increasing **Training Set Size**. Use this visualization to explain why simply collecting more training data will fail to fix a high-bias issue.
6. Beyond changing the dataset size, state one concrete engineering strategy to reduce high bias, and one concrete engineering strategy to reduce high variance.