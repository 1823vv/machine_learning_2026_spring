
# Spiral Classification in TensorFlow Playground

The following experiment uses [TensorFlow Playground](https://playground.tensorflow.org) with the Spiral dataset.

---

## Stage 1 — Raw Features Only

Enable only:

* $x_1$
* $x_2$

Use:

* no hidden layers

A single neuron computes:

$$
z = w_1 x_1 + w_2 x_2 + b
$$

Prediction:

$$
\hat{y} = \sigma(z)
$$

The decision boundary is linear in the input space.

The spiral classes are not separable.

> [!WARNING]
> The main issue is not optimization.
>
> The issue is representation capacity.

---

## Linear Decision Boundary

A linear classifier separates space using a hyperplane.

In 2D, this becomes a line.

The spiral dataset requires a highly nonlinear boundary that repeatedly rotates around the center.

A linear model cannot express this structure.

---

# Stage 2 — Polynomial Feature Engineering

Enable:

* $x_1^2$
* $x_2^2$
* $x_1 x_2$

The feature vector becomes:

$$
\phi(x)
=
(x_1, x_2, x_1^2, x_2^2, x_1 x_2)
$$

Now the classifier computes:

$$
z
=

w_1 x_1
+
w_2 x_2
+
w_3 x_1^2
+
w_4 x_2^2
+
w_5 x_1 x_2
+
b
$$

The decision boundary becomes nonlinear in the original input space.

Classification improves slightly.

---

## Interpretation of Polynomial Features

The polynomial terms introduce curved geometry:

* $x_1^2 + x_2^2$ relates to circular structure
* $x_1 x_2$ introduces rotational interaction

These features help represent:

* circles
* ellipses
* parabolic regions

However, spirals remain difficult because the boundary continuously rotates as radius increases.

---

# Stage 3 — Add Sinusoidal Features

Enable:

* $\sin(x_1)$
* $\sin(x_2)$

Now:

$$
\phi(x)
=
(
x_1,
x_2,
x_1^2,
x_2^2,
x_1 x_2,
\sin(x_1),
\sin(x_2)
)
$$

The sinusoidal terms introduce periodic structure into the feature space.

The boundary becomes more flexible and locally adaptive.

Some spiral regions are classified correctly.

Still not fully separable.

> [!INFO]
> Traditional machine learning often depended heavily on manually engineered features.
>
> Deep learning instead learns useful features automatically.

---

# Stage 4 — Add a Hidden Layer

Add:

* 1 hidden layer
* 6–8 neurons

Each hidden neuron computes:


The hidden layer creates nonlinear feature transformations.

The network begins learning:

* local curves
* angular regions
* nonlinear partitions

The boundary starts wrapping around the spirals.

---

# Stage 5 — Add More Layers

Try:

* 2 hidden layers
* 8 neurons each

or:

* 3 hidden layers

The spiral classes become separable.

The network learns increasingly complex geometric transformations.

---

# Deep Representation Learning

A deep network performs a sequence of transformations:

$$
x
\rightarrow
h^{(1)}
\rightarrow
h^{(2)}
\rightarrow
\cdots
\rightarrow
y
$$

Earlier layers learn simple local patterns.

Later layers combine them into larger global structures.

The network gradually transforms the original space into a representation where separation becomes easier.

---

# Geometric Interpretation

The hidden layers effectively "untwist" the spirals.

In the learned representation space:

* nearby points from the same class move closer together
* different classes become easier to separate

Deep learning can therefore be interpreted as learning new coordinate systems.

---

# Main Insight

Traditional machine learning:

$$
\text{human-designed features}
\rightarrow
\text{classifier}
$$

Deep learning:

$$
\text{learned representations}
\rightarrow
\text{classifier}
$$

The power of deep neural networks comes from hierarchical nonlinear feature learning.
