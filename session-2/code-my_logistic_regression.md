# Logistic Regression from Scratch

## From Linear Regression to Logistic Regression

To understand Logistic Regression, it's helpful to first recap Linear Regression. In previous session, we implemented Linear Regression from scratch:


```python
class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=30000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # Predict the target values
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

The key difference between Linear Regression and Logistic Regression is that Logistic Regression applies a sigmoid function to the linear model output to transform the predictions into probabilities between 0 and 1. This makes Logistic Regression suitable for binary classification problems where we need to predict one of two possible outcomes.


## Implementation: Logistic Regression with Gradient Descent

Let's implement Logistic Regression using Gradient Descent:

```python
import numpy as np

class MyOwnLogisticRegressionGD:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            # Calculate linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
```






### The Math Behind Gradient Descent for Logistic Regression

Under our row-vector convention, each training sample is written as:

$$
x^{(i)} \in \mathbb{R}^{1 \times d}
$$

The Logistic Regression model first computes a linear projection:

$$
z^{(i)} = x^{(i)} W + b
$$

where:

- $x^{(i)} \in \mathbb{R}^{1 \times d}$
- $W \in \mathbb{R}^{d \times 1}$
- $b \in \mathbb{R}^{1 \times 1}$

Thus:

$$
z^{(i)} \in \mathbb{R}^{1 \times 1}
$$

This scalar output is then transformed into a probability using the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

So:

$$
\hat{y}^{(i)} = \sigma(z^{(i)})
$$

where $\hat{y}^{(i)}$ is the predicted probability that sample $i$ belongs to class 1.

To measure how well predictions match true labels, we use Binary Cross-Entropy Loss over all $m$ samples:

$$
\mathcal{L}(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

To perform Gradient Descent, we need:

$$
dW = \frac{\partial \mathcal{L}}{\partial W}
\quad \text{and} \quad
db = \frac{\partial \mathcal{L}}{\partial b}
$$

For one sample, define:

$$
\ell^{(i)} = -\left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

Because the loss depends on $W$ through:

$$
W \rightarrow z^{(i)} \rightarrow \hat{y}^{(i)} \rightarrow \ell^{(i)}
$$

we apply the chain rule:

$$
\frac{\partial \ell^{(i)}}{\partial W} = \frac{\partial \ell^{(i)}}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial W}
$$

**First:**

$$
\frac{\partial \ell^{(i)}}{\partial \hat{y}^{(i)}} = -\left( \frac{y^{(i)}}{\hat{y}^{(i)}} - \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} \right)
$$

**Second**, since the sigmoid derivative is:

$$
\frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} = \hat{y}^{(i)}(1 - \hat{y}^{(i)})
$$

**Third**, because:

$$
z^{(i)} = x^{(i)}W + b
$$

we get:

$$
\frac{\partial z^{(i)}}{\partial W} = x^{(i)\mathsf{T}}
$$

**Substituting:**

$$
\frac{\partial \ell^{(i)}}{\partial W} = -\left( \frac{y^{(i)}}{\hat{y}^{(i)}} - \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} \right) \cdot \hat{y}^{(i)}(1 - \hat{y}^{(i)}) \cdot x^{(i)\mathsf{T}}
$$

After simplification, the complex terms collapse beautifully into:

$$
\frac{\partial \ell^{(i)}}{\partial W} = x^{(i)\mathsf{T}}(\hat{y}^{(i)} - y^{(i)})
$$

Averaging across all samples:

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{n} X^{\mathsf{T}}(\hat{y} - y)
$$

For the bias term, since:

$$
\frac{\partial z^{(i)}}{\partial b} = 1
$$

we similarly obtain:

$$
\frac{\partial \ell^{(i)}}{\partial b} = \hat{y}^{(i)} - y^{(i)}
$$

and over the full dataset:

$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})
$$

This gives the final Gradient Descent updates:

$$
W \leftarrow W - \eta \, dW
$$

$$
b \leftarrow b - \eta \, db
$$

The key insight is that although Logistic Regression combines:

- linear projection
- sigmoid nonlinearity
- logarithmic loss

the final gradient simplifies to:

$$
\hat{y} - y
$$

This elegant simplification is why Sigmoid + Binary Cross-Entropy is such a powerful pairing: mathematically clean, computationally efficient, and ideal for binary classification.



You'll notice that for the derivatives $dW$ and $db$ we have:

$$
dW = \frac{1}{n} X^{\mathsf{T}} \cdot (\hat{y} - y)
$$

$$
db = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})
$$

These formulas look remarkably similar to the ones used for Linear Regression, which is a **beautiful** result of using cross-entropy loss with the sigmoid activation function.



## Testing the Gradient Descent Implementation

Let's test our implementation on a simple classification dataset:

```python
# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a simple classification dataset
X, y = datasets.make_classification(
    n_samples=100, n_features=2, n_redundant=0, 
    n_informative=2, random_state=1, n_clusters_per_class=1
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the logistic regression model
model = MyOwnLogisticRegressionGD(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Visualize the results
def plot_decision_boundary(X, y, model):
    # Define the bounds of the plot
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Make predictions on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create a contour plot
    plt.contourf(xx, yy, Z, alpha=0.3)
    
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50)
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot the decision boundary
plot_decision_boundary(X, y, model)
```

You should be able to see the decision boundary is a line, and has math formulation:

$$w_1 x_1 + w_2 x_2 + b = 0$$

which is equivalent to:

$$x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}$$
