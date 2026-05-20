## Question: TensorFlow Playground Feature Engineering for Curved and Ring-Shaped Boundaries

In a live TensorFlow Playground demo, we compare logistic regression behavior under different feature sets for binary classification with two raw inputs $(x_1, x_2)$. We focus on curved geometry such as center-vs-ring patterns, where class $y=0$ is concentrated near the origin and class $y=1$ forms an outer ring.

1. Starting from raw features only, write the logistic regression score $z$ and the decision boundary equation ($z=0$). Explain geometrically why this setup underfits the ring-vs-center pattern.
2. For univariate regression, write degree-2 and degree-3 polynomial models, then the general degree-$k$ form. Define a feature map $\phi(x)$ and explain why polynomial regression is still linear in parameters during optimization.
3. In TensorFlow Playground, enable engineered terms $x_1^2$ and $x_2^2$ so that $\phi(x) = [x_1, x_2, x_1^2, x_2^2]^T$. Write the transformed logistic score and boundary. State the qualitative sign pattern for weights/bias that would make $P(y=1\mid x)$ small near $(0,0)$ and larger in outer regions.
4. Suppose the class geometry changes from concentric circles to (i) axis-aligned ellipses and (ii) rotated ellipses. For each case, identify which engineered terms are required and explain when the cross term $x_1x_2$ becomes necessary.
5. Sketch three TensorFlow-Playground-style boundary outcomes on the original ring-vs-center dataset: (i) degree-1 underfitting, (ii) degree-2 good fit, and (iii) overly high-degree overfitting. Briefly interpret the bias-variance trade-off across the three sketches.
