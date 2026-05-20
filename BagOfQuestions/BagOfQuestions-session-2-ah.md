## Question: Two Spirals in TensorFlow Playground and Representation Learning

Consider binary classification in TensorFlow Playground with the **Spiral** dataset, raw inputs $(x_1, x_2)$, and sigmoid output probability $\hat y=\sigma(z)$.

1. **Stage 1 (Raw features, no hidden layer).** Use only $x_1, x_2$ and no hidden layers. Write the model score
   $$
   z = w_1x_1 + w_2x_2 + b,
   $$
   then explain why this boundary class cannot separate two intertwined spirals.
2. **Stage 2 (Manual feature engineering).** Add polynomial terms $x_1^2, x_2^2, x_1x_2$, and optionally sinusoidal terms $\sin(x_1), \sin(x_2)$. Write an explicit transformed feature map $\phi(x)$ and a linear-in-parameters score $z=w^T\phi(x)+b$. Explain the expected effect: partial improvement and richer curvature, but still limited global spiral separation.
3. **Stage 3 (Add depth).** Keep the same input features and add hidden layers with ReLu activations, for example one hidden layer (6--8 neurons), then two or three layers (about 8--12 neurons each). Explain how compositions
   $$
   x \rightarrow h^{(1)} \rightarrow h^{(2)} \rightarrow \cdots \rightarrow \hat y
   $$
   can be interpreted as progressively transforming coordinates so classes become easier to separate.
4. Draw and label a four-panel progression of decision boundaries for: (i) raw logistic model, (ii) engineered features only, (iii) shallow network, and (iv) deeper network. For each panel, add one sentence describing what geometric structure is captured and what remains incorrect.
5. Write a short concluding paragraph (3--5 sentences): Why is the two-spirals experiment a strong argument for deep learning, and why is “deep works because of learned representations” a stronger explanation than “deep works because it has many parameters”?
