# Linear Regression from Scratch using TDD & PyTest

In this session, you will:

- Implement your own **Linear Regression** using only NumPy (no `sklearn`).
- Follow **Test‑Driven Development** (Red → Green → Refactor).
- Write **PyTest tests** that use randomly generated data.
- Run tests locally and set up **GitHub Actions** for CI/CD.

---

## Step 1: Project Setup

Create the following structure:

```
linear_regression_tdd/
├── requirements.txt
├── linear_regression.py          # your implementation
├── tests/
│   └── test_linear_regression.py
└── .github/
    └── workflows/
        └── test.yml
```

### 1.1 `requirements.txt`

```text
pytest
numpy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 2: Write a Failing Test (Red)

In `tests/test_linear_regression.py`, write a test that uses randomly generated linear data and checks that the model can learn approximately correct weights.

```python
# tests/test_linear_regression.py
import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from linear_regression import MyOwnLinearRegression

def test_linear_regression_on_random_data():
    # Generate synthetic data: y = 3*x + 2 + noise
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    true_weight = 3.0
    true_bias = 2.0
    y = true_weight * X.flatten() + true_bias + np.random.randn(100) * 0.5

    model = MyOwnLinearRegression(learning_rate=0.01, n_iters=2000)
    model.fit(X, y)

    # After training, weight should be close to 3, bias close to 2
    assert abs(model.weights[0] - 3.0) < 0.3
    assert abs(model.bias - 2.0) < 0.3
```

Run the test:

```bash
pytest tests/test_linear_regression.py -v
```

**Expected:** `ModuleNotFoundError` or `AttributeError` – because `MyOwnLinearRegression` does not exist yet.  
👉 **Red** step.

---

## Step 3: Implement the Skeleton to Pass (Green)

Create `linear_regression.py` with the following **blanks** (`______YOUR_CODE_HERE_________`).  
Fill them to make the test pass.

```python
# linear_regression.py
import numpy as np

class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=10000):
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
            # 1. Predict y using current weights and bias
            y_predicted = ______YOUR_CODE_HERE_________

            # 2. Compute gradients (MSE derivatives)
            dw = ______YOUR_CODE_HERE_________
            db = ______YOUR_CODE_HERE_________

            # 3. Update parameters
            self.weights = ______YOUR_CODE_HERE_________
            self.bias = ______YOUR_CODE_HERE_________

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

**Fill in the blanks** (answers below – but try yourself first):

```python
y_predicted = np.dot(X, self.weights) + self.bias
dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
db = (2 / n_samples) * np.sum(y_predicted - y)
self.weights = self.weights - self.lr * dw
self.bias = self.bias - self.lr * db
```

Now run the test again – it should pass (Green).

---

## Step 4: Add a Second Test – Prediction Shape

Add another test to verify that `predict()` returns an array of the correct shape.

In `tests/test_linear_regression.py`:

```python
def test_predict_returns_correct_shape():
    model = MyOwnLinearRegression()
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
```

Run both tests – they should pass.

---

## Step 5: Refactor (Optional)

You could extract the gradient computation into a separate method, but for a simple exercise this is fine.  
The key is that **tests stay green** after refactoring.

---

## Step 6: Run All Tests Locally

```bash
pytest -v
```

You should see output like:

```
tests/test_linear_regression.py::test_linear_regression_on_random_data PASSED
tests/test_linear_regression.py::test_predict_returns_correct_shape PASSED
```

---

## Step 7: CI/CD with GitHub Actions

Create `.github/workflows/test.yml`:

```yaml
name: Test Linear Regression

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest -v
```

Push your repository to GitHub. The workflow will run automatically on every push or pull request.

---

## Full Final Code

### `linear_regression.py` (completed)

```python
import numpy as np

class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

### `tests/test_linear_regression.py` (completed)

```python
import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from linear_regression import MyOwnLinearRegression

def test_linear_regression_on_random_data():
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    true_weight = 3.0
    true_bias = 2.0
    y = true_weight * X.flatten() + true_bias + np.random.randn(100) * 0.5

    model = MyOwnLinearRegression(learning_rate=0.01, n_iters=2000)
    model.fit(X, y)

    assert abs(model.weights[0] - 3.0) < 0.3
    assert abs(model.bias - 2.0) < 0.3

def test_predict_returns_correct_shape():
    model = MyOwnLinearRegression()
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
```

---

## ✅ TDD Takeaways for This Exercise

1. **Red** – Wrote a test that used a non‑existent class – it failed.
2. **Green** – Implemented the minimal code (with blanks filled) – test passed.
3. **Refactor** – Improved code structure while keeping tests green.
4. **CI/CD** – Automated testing on every push.

You have just built and tested your own linear regression model from scratch using TDD!