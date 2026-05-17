# Logistic Regression from Scratch using TDD & PyTest

In this session, you will:

- Implement your own **Logistic Regression** using only NumPy (no `sklearn`).
- Follow **Test‑Driven Development** (Red → Green → Refactor).
- Write **PyTest tests** with randomly generated binary classification data.
- Run tests locally and set up **GitHub Actions** for CI/CD.

---

## Step 1: Project Setup

Create the following structure:

```
logistic_regression_tdd/
├── requirements.txt
├── logistic_regression.py          # your implementation
├── tests/
│   └── test_logistic_regression.py
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

In `tests/test_logistic_regression.py`, write a test that uses randomly generated linearly separable data and checks that the model can achieve high accuracy.

```python
# tests/test_logistic_regression.py
import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from logistic_regression import MyOwnLogisticRegression

def test_logistic_regression_on_synthetic_data():
    # Generate synthetic binary data: two clusters
    np.random.seed(42)
    n_samples = 200
    X_class0 = np.random.randn(n_samples // 2, 2) + [1, 1]
    X_class1 = np.random.randn(n_samples // 2, 2) + [3, 3]
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    model = MyOwnLogisticRegression(learning_rate=0.1, n_iters=2000)
    model.fit(X, y)
    predictions = model.predict(X)

    accuracy = np.mean(predictions == y)
    assert accuracy > 0.95
```

Run the test:

```bash
pytest tests/test_logistic_regression.py -v
```

**Expected:** `ModuleNotFoundError` or `AttributeError` – because `MyOwnLogisticRegression` does not exist yet.  
👉 **Red** step.

---

## Step 3: Implement the Skeleton to Pass (Green)

Create `logistic_regression.py` with the following **blanks** (`______YOUR_CODE_HERE_________`).  
Fill them to make the test pass.

```python
# logistic_regression.py
import numpy as np

class MyOwnLogisticRegression:
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
            # Linear combination
            linear_model = ______YOUR_CODE_HERE_________
            # Apply sigmoid
            y_predicted = ______YOUR_CODE_HERE_________

            # Compute gradients (log loss / binary cross-entropy derivatives)
            dw = ______YOUR_CODE_HERE_________
            db = ______YOUR_CODE_HERE_________

            # Update parameters
            self.weights = ______YOUR_CODE_HERE_________
            self.bias = ______YOUR_CODE_HERE_________

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        # Convert probabilities to class labels (threshold = 0.5)
        y_predicted_cls = ______YOUR_CODE_HERE_________
        return np.array(y_predicted_cls)

    def _sigmoid(self, z):
        return ______YOUR_CODE_HERE_________
```

**Fill in the blanks**. Now run the test again – it should pass (Green).

---

## Step 4: Add a Second Test – Prediction Shape

Add another test to verify that `predict()` returns an array of the correct shape.

In `tests/test_logistic_regression.py`:

```python
def test_predict_returns_correct_shape():
    model = MyOwnLogisticRegression()
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([0, 1, 0])
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
```

Run both tests – they should pass.

---

## Step 5: Refactor (Optional)

You could move the gradient computation into a separate method or add a `score()` method that returns accuracy.  
As long as tests stay green, any refactoring is safe.

---

## Step 6: Run All Tests Locally

```bash
pytest -v
```

Expected output:

```
tests/test_logistic_regression.py::test_logistic_regression_on_synthetic_data PASSED
tests/test_logistic_regression.py::test_predict_returns_correct_shape PASSED
```

---

## Step 7: CI/CD with GitHub Actions

Create `.github/workflows/test.yml`:

```yaml
name: Test Logistic Regression

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

### `logistic_regression.py` (completed)

```python
import numpy as np

class MyOwnLogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return np.array([1 if i > 0.5 else 0 for i in y_predicted])

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
```

### `tests/test_logistic_regression.py` (completed)

```python
import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from logistic_regression import MyOwnLogisticRegression

def test_logistic_regression_on_synthetic_data():
    np.random.seed(42)
    n_samples = 200
    X_class0 = np.random.randn(n_samples // 2, 2) + [1, 1]
    X_class1 = np.random.randn(n_samples // 2, 2) + [3, 3]
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    model = MyOwnLogisticRegression(learning_rate=0.1, n_iters=2000)
    model.fit(X, y)
    predictions = model.predict(X)

    accuracy = np.mean(predictions == y)
    assert accuracy > 0.95

def test_predict_returns_correct_shape():
    model = MyOwnLogisticRegression()
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([0, 1, 0])
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
```

---

## ✅ TDD Takeaways for Logistic Regression

- **Red** – Wrote a test that uses a non‑existent class – it failed.
- **Green** – Implemented the `fit`, `predict`, and `_sigmoid` methods (filling the blanks) – test passed.
- **Refactor** – Kept code clean while ensuring tests remain green.
- **CI/CD** – Automated testing on every push guarantees the model never regresses.

You have now implemented and tested your own logistic regression model from scratch using TDD!