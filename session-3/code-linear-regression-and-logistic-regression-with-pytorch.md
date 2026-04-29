# Linear & Logistic Regression with PyTorch

---

# 1. Why PyTorch?

In previous sessions we implemented:

* **Linear Regression**
* **Logistic Regression**

using **scikit-learn**.

Example:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
```

This is extremely convenient.

However, there is an important limitation:

> We do not see **how the model is trained internally**.

Libraries like **scikit-learn** hide the training procedure.

Frameworks like **PyTorch** expose the full training process.
They allow us to:

* define models
* compute predictions
* compute loss
* update parameters with gradient descent

Understanding this process is essential because it is **exactly how deep learning models are trained**.

---

# 2. Key Idea of This Session

The key conceptual message today is:

> **Linear Regression is a Neural Network**

> **Logistic Regression is also a Neural Network**

More precisely:

| Model               | Neural Network Interpretation |
| ------------------- | ----------------------------- |
| Linear Regression   | 1 Linear Layer                |
| Logistic Regression | 1 Linear Layer + Sigmoid      |

---

### Linear Regression Structure

```
x1 ──\
x2 ───> Linear Layer → y
x3 ──/
```

---

### Logistic Regression Structure

```
x1 ──\
x2 ───> Linear Layer → Sigmoid → probability
x3 ──/
```

So in today's session we will implement both models using **PyTorch**.

This will allow us to see the **complete training pipeline** used in modern deep learning.

---

# 4. Linear Regression with sklearn

First, let's recall the **scikit-learn implementation**.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
```

Although the code looks simple, internally several steps MIGHT occur:

1. Initialize weights
2. Compute predictions
3. Compute loss
4. Update weights using gradient descent
5. Repeat many times

But **scikit-learn hides these steps from us**.

PyTorch will allow us to implement them explicitly.

---

# 5. Linear Regression with PyTorch

Now we implement the **same model using PyTorch**.

---

## Step 1 — Import libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

Explanation:

* **torch** → core tensor library
* **torch.nn** → neural network layers
* **torch.optim** → optimization algorithms

---

## Step 2 — Convert data to tensors

PyTorch does not operate on NumPy arrays directly.

Instead, it uses **tensors**.

```python
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
```

Key detail:

```
view(-1,1)
```

reshapes the target variable into a **column vector**, which PyTorch expects.

---

## Step 3 — Define the model

This is the most important step.

```python
model = nn.Linear(in_features=1, out_features=1)
```

Mathematically this layer represents:

```
y = w*x + b
```

Which is **exactly the equation of Linear Regression**.

So we have just created a **one-layer neural network**.

---

## Step 4 — Define loss function

For regression problems we typically use **Mean Squared Error (MSE)**.

```python
criterion = nn.MSELoss()
```

The loss function measures:

```
difference between predictions and true values
```

---

## Step 5 — Define optimizer

We use **Stochastic Gradient Descent (SGD)** to update the parameters.

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

Here:

* `model.parameters()` → weights and bias
* `lr` → learning rate

---

## Step 6 — Training loop

Now we implement the **training process**.

```python
epochs = 200

for epoch in range(epochs):

    predictions = model(X_tensor)

    loss = criterion(predictions, y_tensor)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 20 == 0:
        print(epoch, loss.item())
```

The training workflow follows the standard deep learning pipeline:

```
Forward pass
      ↓
Compute loss
      ↓
Backward pass (compute gradients)
      ↓
Update parameters
```

This pipeline is used in **all modern neural networks**.

---

# 6. Visualizing the Result

We can visualize the fitted regression line.

```python
import matplotlib.pyplot as plt

with torch.no_grad():
    predictions = model(X_tensor)

plt.scatter(X, y)
plt.plot(X, predictions.numpy(), color="red")
plt.show()
```

This plot shows:

* the original data points
* the regression line learned by the model

---

# 7. Logistic Regression with sklearn

Now we move to **Logistic Regression**.

First, let's review the **scikit-learn version**.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X, y)

predictions = model.predict(X)
```

Here the model outputs **class labels**.

But internally it actually predicts **probabilities**.

---

# 8. Logistic Regression with PyTorch

Now we implement **Logistic Regression using PyTorch**.

Mathematically:

```
z = w₁x₁ + w₂x₂ + b
p = sigmoid(z)
```

Where the **sigmoid function** is:

```
sigmoid(z) = 1 / (1 + e^-z)
```

The output **p** represents:

```
probability that the sample belongs to class 1
```

---

### Network structure

```
x1 ──\
      Linear → Sigmoid → probability
x2 ──/
```

This is again a **single-layer neural network**.

---

# 9. Generate a Classification Dataset

We create a synthetic dataset for classification.

```python
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    flip_y=0.1,  
    class_sep=0.8,  
    n_clusters_per_class=1,
    random_state=45,
)
```

This dataset has **two features**, which allows us to visualize the classification boundary.

---

### Visualizing the dataset

```python
import matplotlib.pyplot as plt

plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Synthetic Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

Students should see **two clusters of points** corresponding to the two classes.

---

# Logistic Regression with sklearn (Visualization)

Before implementing Logistic Regression with PyTorch, let's first train a **scikit-learn Logistic Regression model** on the same dataset and visualize its decision boundary.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X, y)

predictions = model.predict(X)
```

Now we can visualize the predictions.

```python
plt.scatter(X[:,0], X[:,1], c=predictions)
plt.title("Logistic Regression (sklearn) Predictions")
plt.show()
```

To visualize the decision boundary we use the same **grid technique**.

```python
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
    np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]

probs = model.predict_proba(grid)[:,1]

Z = probs.reshape(xx.shape)
```

Plot the decision boundary.

```python
plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(X[:,0], X[:,1], c=y)

plt.title("Decision Boundary (sklearn Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
```

Now we have a **reference model** implemented with sklearn.

Next we will implement the same model using **PyTorch**.

---

# 10. Convert Data to PyTorch Tensors

```python
import torch

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1,1)
```

Again we convert the data to **PyTorch tensors**.

---

# 11. Define the Logistic Regression Model

Logistic Regression can be implemented as:

```
Linear Layer
+ Sigmoid
```

Implementation:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()
)
```

This already defines a **neural network**.

---

# 12. Loss Function

For binary classification we use **Binary Cross Entropy**.

```python
criterion = nn.BCELoss()
```

This loss measures the difference between:

```
predicted probability
vs
true label
```

---

# 13. Optimizer

We again use **Stochastic Gradient Descent**.

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.1)
```

---

# 14. Training Loop

Now we train the model.

```python
epochs = 300

for epoch in range(epochs):

    preds = model(X_tensor)

    loss = criterion(preds, y_tensor)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print(epoch, loss.item())
```

During training, the model learns a **decision boundary** separating the two classes.

---

# 15. Convert Probabilities to Classes

The model outputs **probabilities**, not class labels.

Example:

```
0.87 → class 1
0.21 → class 0
```

We apply a threshold of **0.5**.

```python
with torch.no_grad():

    probs = model(X_tensor)

    predicted = (probs > 0.5).float()
```

---

# 16. Visualizing the Predictions

Now we plot the predicted labels.

```python
plt.scatter(X[:,0], X[:,1], c=predicted.numpy())
plt.title("Model Predictions")
plt.show()
```

If training succeeded, the model should correctly separate the clusters.

---

# 17. Visualizing the Decision Boundary

To visualize the decision boundary we create a **grid of points**.

```python
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
    np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]

grid_tensor = torch.tensor(grid, dtype=torch.float32)
```

Now we ask the model to classify every point on the grid.

```python
with torch.no_grad():
    probs = model(grid_tensor)

Z = probs.numpy().reshape(xx.shape)
```

Plot the decision boundary.

```python
plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(X[:,0], X[:,1], c=y)

plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
```

Students will now see a **smooth boundary separating the two classes**.

---

# 18. Visualizing the Probability Surface

We can also visualize probability levels.

```python
plt.contour(xx, yy, Z, levels=[0.1,0.3,0.5,0.7,0.9])

plt.scatter(X[:,0], X[:,1], c=y)

plt.title("Probability Contours")
plt.show()
```

Key insight:

```
P = 0.5
```

represents the **decision boundary**.

---

# 19. Inspecting the Model Parameters

Finally, we inspect the learned parameters.

```python
print(model[0].weight)
print(model[0].bias)
```

This corresponds exactly to:

```
z = w₁x₁ + w₂x₂ + b
```

The decision boundary is defined by:

```
w₁x₁ + w₂x₂ + b = 0
```

Which represents a **straight line in 2D space**.
