# Core Terminology of Machine Learning

---

## 1. Inputs and Data

A machine learning model starts with data:

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^n
$$

### Key Terms

* **Input / Feature / Variable**:
  $x \in \mathbb{R}^{1 \times d}$ — one input row vector used to make predictions
* **Feature Vector**:
  A collection of features, e.g. $x = (x_1, \dots, x_d)$, written as a row vector in these sessions
* **Sample / Instance / Observation**:
  One data point $(x^{(i)}, y^{(i)})$

---

## 2. Outputs and Predictions

Each input has a corresponding target:

$$
y \quad \text{and} \quad \hat{y} = f(x)
$$

### Key Terms

* **Target / Label / Ground Truth ($y$)**
  The true value we want to learn

* **Prediction / Output / Estimate ($\hat{y}$)**
  The model’s output

---

## 3. Model and Parameters

The model defines how inputs map to outputs. In general notation we may write:

$$
\hat{y} = f(x; \theta)
$$

For neural-network layer formulas, we use the row-vector convention:

$$
z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}
$$

### Key Terms

* **Model / Function / Hypothesis**
  The mapping from $x$ to $\hat{y}$

* **Parameters ($\theta$)**
  A compact name for all learnable components. In neural-network formulas, these are usually written as layer weights $W^{(l)}$ and biases $b^{(l)}$.

* **Weights**
  Parameters in neural networks

* **Coefficients**
  Parameters in linear regression

* **Bias / Intercept ($b$)**
  Constant offset in the model

---

## 4. Neural Network Terminology


* **Neural Network (NN) / Artificial Neural Network (ANN)**
  A model composed of multiple layers of parameterized functions

* **Feedforward Network (FFN)**
  A network where information flows strictly from input to output (no cycles)

* **Multilayer Perceptron (MLP)**
  A standard fully-connected feedforward neural network

* **Layer**
  A transformation of the form:
  $$
  z = x W + b, \quad h = \sigma(z)
  $$

* **Hidden Layer**
  Intermediate layers between input and output

* **Activation Function**
  Nonlinear function such as $\sigma(\cdot)$, ReLU, etc.

* **Forward Pass**
  The computation of $\hat{y}$ from $x$ through the network

---

## 5. Loss and Optimization

Learning is defined as minimizing error:

$$
\min_{W,b} \; \mathcal{L}(\hat{y}, y)
$$

### Key Terms

* **Error**:
  Difference between prediction and truth ($\hat{y} - y$)

* **Loss Function/Cost / Objective**:
  Measures how bad the prediction is (aggregated over the dataset)

* **Optimization**:
  The process of finding best parameters

* **Gradient Descent**:
  A method to update parameters:
  $$
  W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}
  $$

---

### Final Insight

> Machine learning is the process of learning parameters such as $W$ and $b$
> so that a function $f(x; W,b)$ produces predictions $\hat{y}$
> that match targets $y$ by minimizing a loss,
> while generalizing to new data.
