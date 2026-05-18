# L1 vs L2 Regularization — Geometry of Norm Balls and the First-Touch Principle

## 1. Conceptual Overview

L1 and L2 regularization are best understood not as algebraic penalties, but as geometric constraints interacting with the structure of the loss function.

The key object is the interaction between:

* the geometry of the **loss function level sets**
* the geometry of the **norm constraint region**

The solution is determined by a single geometric event:

> the first point of contact between a loss level set and the constraint boundary

This is the core reason L1 produces sparsity while L2 produces smooth shrinkage.

---

## 2. From Penalized Form to Constrained Form

Regularized optimization is written as:

$$
\min_W \mathcal{L}_{\text{train}}(W) + \lambda \Omega(W)
$$

This is equivalent to the constrained optimization problem:

$$
\min_W \mathcal{L}_{\text{train}}(W) \quad \text{s.t.} \quad \Omega(W) \leq c
$$

The parameter $\lambda$ controls the tightness of the constraint $c$.

This equivalence is essential because it transforms regularization into a geometric problem.

---

## 3. Geometry of the Loss Function

To understand the “first-touch” phenomenon, we examine the shape of the loss landscape.

Near an optimum, the loss can be approximated by a quadratic form:

$$
\mathcal{L}(W) \approx (W - W^*)^T H (W - W^*)
$$

where:

* $W^*$ is the unconstrained minimizer
* $H$ is the Hessian matrix encoding curvature

---

### 3.1 Level sets of the loss

A level set is defined as:

$$
\mathcal{L}(W) = \text{constant}
$$

In two dimensions, these level sets are ellipses.

In higher dimensions, they become ellipsoids.

Each level set corresponds to a fixed value of the loss.

---

### 3.2 Geometric meaning

* inner ellipses correspond to lower loss values
* outer ellipses correspond to higher loss values
* the center corresponds to the unconstrained optimum $W^*$

The optimization problem becomes a search over nested geometric surfaces.

---

## 4. “First Touch”

The solution is the lowest loss level set that intersects the feasible region.

* start at the unconstrained optimum $W^*$
* consider level sets of increasing loss
* ellipses expand outward
* the solution is the first level set that hits the constraint boundary


---

## 5. L2 Regularization: Circular Geometry

### 5.1 Constraint definition

L2 regularization defines:

$$
\sum_{j=1}^{d} W_j^2 \leq c
$$

In two dimensions:

$$
w_1^2 + w_2^2 \leq c
$$

This forms a circle (or hypersphere in higher dimensions).

---

### 5.2 Geometric properties

The L2 constraint has:

* smooth boundary everywhere
* no corners or edges
* rotational symmetry in all directions

All coordinate directions are treated equally.

---

### 5.3 First-touch behavior

When elliptical loss contours intersect a circle:

* the contact point is typically smooth
* no axis is preferred
* all coordinates tend to remain non-zero

---

### 5.4 Optimization consequence for L2

The gradient condition at optimum is:

$$
\nabla \mathcal{L}(W) = -\lambda W
$$

This implies proportional shrinkage of parameters.

The result is:

* smooth reduction of all weights
* rare exact zeros
* dense solutions

---

## 6. L1 Regularization: Diamond Geometry

### 6.1 Constraint definition

L1 regularization defines:

$$
\sum_{j=1}^{d} |W_j| \leq c
$$

In two dimensions:

$$
|w_1| + |w_2| \leq c
$$

This forms a diamond-shaped polytope.

---

### 6.2 Geometric structure

The boundary consists of:

* flat edges
* sharp corners
* axis-aligned structure

Corner points include:

* $(c, 0)$
* $(0, c)$
* $(-c, 0)$
* $(0, -c)$

---

### 6.3 First-touch behavior

When ellipses expand outward:

* they often intersect the boundary at corners first
* corners correspond to coordinate axes
* at a corner, at least one coordinate is exactly zero

This directly induces sparsity.

---

### 6.4 Why corners dominate

Corners are geometrically extreme points:

* they maximize one coordinate under a linear constraint
* they lie on coordinate axes
* they represent degenerate but optimal trade-offs

This makes sparse solutions geometrically favorable.

---

## 7. Subgradient Structure of L1

L1 is non-smooth at zero.

The derivative is:

$$
\frac{\partial |w_j|}{\partial w_j} =
\begin{cases}
1 & w_j > 0 \\
-1 & w_j < 0 \\
[-1, 1] & w_j = 0
\end{cases}
$$

---

### 7.1 Key implication

At $w_j = 0$, the gradient is a set, not a single value.

This allows:

* stable zero solutions
* threshold-like behavior during optimization
* persistence of sparsity once achieved

---

## 8. High-Dimensional Geometry

### 8.1 L2 geometry

$$
\sum_{j=1}^{d} W_j^2 \leq c
$$

* hypersphere
* smooth curvature
* isotropic structure

No coordinate direction is special.

---

### 8.2 L1 geometry

$$
\sum_{j=1}^{d} |W_j| \leq c
$$

* cross-polytope
* many corners
* axis-aligned structure

---

### 8.3 Dimensional effect

As dimension increases:

* L1 produces exponentially many corners
* probability of hitting sparse vertices increases
* sparsity becomes more dominant in high dimensions

---

## 9. Unified Geometric Principle

The full picture can be summarized as:

* loss defines elliptical level sets
* regularization defines a feasible geometric region
* solution is determined at their first intersection

Different geometries encode different inductive biases:

* L2: smooth convex boundary → distributed shrinkage
* L1: polyhedral boundary → sparsity via corners
