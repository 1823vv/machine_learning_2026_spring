## Question: Polynomial Feature Engineering and Non-linear Decision Boundaries

A linear model learns a straight-line hyperplane boundary when using raw features. If a dataset shows a curved or enclosed relationship, a basic linear model will underfit. We can resolve this by applying a non-linear feature transformation $\phi(x)$ to map our raw inputs into a higher-dimensional space. This allows a linear optimizer to successfully separate the patterns.

### Part A: From Univariate Regression to Multiple Linear Regression

1. Suppose a dataset shows a curved relationship between a single input $x$ and a continuous target $y$. Write out the explicit equations for a **degree-2** and a **degree-3** polynomial regression model.
2. For a general polynomial regression model of **degree $k$** with a single input $x$, write out the full mathematical model equation. Define its corresponding feature transformation vector $\phi(x)$.
3. Explain why polynomial regression is mathematically considered a form of **multiple linear regression** after the transformation $\phi(x)$ is applied. How can a model be highly non-linear with respect to the raw input $x$, yet remain strictly linear during optimization and parameter updates?

### Part B: Bivariate Spatial Separation via Logistic Regression (The TensorFlow Playground Case)

Imagine a binary classification dataset in the **TensorFlow Playground** with two raw input features, $x_1$ and $x_2$. The data is distributed such that the **Orange class** ($y=0$) forms a concentrated circular cluster centered at the origin $(0,0)$, while the **Blue class** ($y=1$) forms a large concentric ring enclosing the orange cluster. We use **logistic regression** to model the probabilities.

4. Write the mathematical equation for a standard **logistic regression** decision boundary using only the raw features $x_1$ and $x_2$ (where the log-odds equal 0). Explain geometrically why this raw logistic regression model fails completely on this circle-and-ring dataset.
5. Suppose we engineer a quadratic feature transformation mapping the 2D input to a 4D space: $\phi(x) = [x_1, x_2, x_1^2, x_2^2]^T$.
* Write out the modified logistic regression decision boundary equation ($z = 0$) using these transformed features.
* State the qualitative signs (positive, negative, or zero) that the weights and bias must take so that the predicted probability $P(y=1|x)$ is small at the origin and large in the outer ring.


6. The TensorFlow Playground also offers interactive cross-product terms like $x_1x_2$. Explain how your feature requirements for the logistic regression model would change if the geometric pattern mutated:
* **Case A:** The orange and blue clusters are concentric but stretched into ellipses aligned with the coordinate axes. Which specific features are needed?
* **Case B:** The elliptical clusters are rotated and tilted at an angle across the quadrants. Explain why adding the cross-product feature $x_1x_2$ becomes necessary for logistic regression to capture this tilt.



### Part C: Visualizing Complexity

7. Sketch three separate diagrams showing how the logistic regression decision boundary behaves on the original circle-and-ring dataset under different feature configurations:
* **Diagram 1 (Underfitting):** Using raw features ($\text{Degree } 1$).
* **Diagram 2 (Good Fit):** Using the exact required quadratic transformation ($\text{Degree } 2$).
* **Diagram 3 (Overfitting):** Using an excessively complex transformation ($\text{e.g., Degree } 6$).

