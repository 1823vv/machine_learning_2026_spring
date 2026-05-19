# Logistic Regression — A Probabilistic Perspective

## 1. From Optimization to Probability

We have already seen logistic regression as:

* a model using the sigmoid function
* trained with binary cross-entropy (BCE)
* optimized via GD 

Now we reinterpret everything from a probabilistic viewpoint:

> Logistic regression is **maximum likelihood estimation under a Bernoulli model**.

This perspective explains *why* BCE appears and what the model is actually learning.

---

## 2. The Data-Generating Assumption

We consider binary labels:

$$
y \in {0, 1}
$$

We assume:

$$
y \mid x \sim \text{Bernoulli}(p(x))
$$

This means:

$$
p(y=1 \mid x) = p(x), \quad p(y=0 \mid x) = 1 - p(x)
$$

The key modeling task is to define $p(x)$.

---

## 3. From Linear Scores to Probabilities

We start with a linear score:

$$
z = w^\top x
$$

To convert this into a probability, we use the sigmoid function:

$$
p(x) = \sigma(w^\top x) = \frac{1}{1 + e^{-w^\top x}}
$$

Thus:

$$
p(y=1 \mid x) = \sigma(w^\top x)
$$

$$
p(y=0 \mid x) = 1 - \sigma(w^\top x)
$$

---

## 4. Log-Odds Interpretation

A central property of logistic regression is:

$$
\log \frac{p(y=1 \mid x)}{p(y=0 \mid x)} = w^\top x
$$

So the model is linear in **log-odds space**.

This explains why the decision boundary is linear.

---

## 5. Likelihood of One Sample

From the Bernoulli assumption:

$$
p(y \mid x, w) = \sigma(w^\top x)^y (1 - \sigma(w^\top x))^{1-y}
$$

This compact form handles both cases:

* $y=1$ → $\sigma(w^\top x)$
* $y=0$ → $1 - \sigma(w^\top x)$

---

## 6. Likelihood of the Dataset

For i.i.d. data ${(x_i, y_i)}_{i=1}^N$:

$$
p(\mathcal{D} \mid w) = \prod_{i=1}^N \sigma(w^\top x_i)^{y_i}
(1 - \sigma(w^\top x_i))^{1-y_i}
$$

---

## 7. Log-Likelihood

Taking logarithms:

$$
\log p(\mathcal{D} \mid w) = \sum_{i=1}^N
\left[
y_i \log \sigma(w^\top x_i)
+
(1 - y_i)\log(1 - \sigma(w^\top x_i))
\right]
$$

---

## 8. Maximum Likelihood Estimation

We estimate parameters by:

$$
\max_w \log p(\mathcal{D} \mid w)
$$

Equivalently:

$$
\min_w -\log p(\mathcal{D} \mid w)
$$

---

## 9. Connection to BCE

Define $\hat{y}_i = \sigma(w^\top x_i)$.

Then:

$$
-\log p(\mathcal{D} \mid w) = -\sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i) \right]
$$

This is exactly **binary cross-entropy**.

> BCE is the **negative log-likelihood of a Bernoulli model**.

---

## 10. Why No Closed-Form Solution?

In linear regression, Gaussian assumptions lead to a quadratic objective.

Here, the sigmoid introduces nonlinearity:

* the objective is convex but not quadratic
* no closed-form solution exists

Thus, we rely on iterative optimization (e.g. GD).

---

## 11. MAP Estimation and Regularization

Introduce a Gaussian prior:

$$
w \sim \mathcal{N}(0, \lambda^{-1} I)
$$

Then:

$$
\max_w \log p(w \mid \mathcal{D})
$$

is equivalent to:

$$
\min_w
\left[
-\log p(\mathcal{D} \mid w) + \lambda |w|^2
\right]
$$

This gives **L2-regularized logistic regression**.

---

## 12. Relation to Linear Regression

Both models follow the same pattern:

* Linear Regression

  * Gaussian noise
  * MSE loss
  * closed-form solution

* Logistic Regression

  * Bernoulli likelihood
  * BCE loss
  * iterative optimization

> Different probabilistic assumptions lead to different loss functions.

