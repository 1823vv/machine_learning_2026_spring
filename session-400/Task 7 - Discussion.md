# The Mathematics of Debiasing in Word Embeddings

## Introduction

Modern Natural Language Processing systems represent words as vectors in high-dimensional spaces. These vector representations, called word embeddings, allow neural networks and machine learning systems to reason about semantic relationships between words.

However, because embeddings are trained on human-generated text, they often inherit social biases present in the training data. For example:

* "doctor" may become closer to "man"
* "nurse" may become closer to "woman"

This lecture explores the mathematical foundations of debiasing word vectors using orthogonal projection methods from linear algebra.

The core mathematical ideas include:

* Vector spaces
* Dot products
* Orthogonal decomposition
* Projection matrices
* Subspaces and orthogonal complements

---

# Word Embeddings as Geometric Objects

## Vector Representation of Words

In word embedding models such as:

* Word2Vec
* GloVe
* FastText

every word is represented by a vector:

$$
\vec v \in \mathbb R^n
$$

where:

* $n$ may be 50, 100, 300, or larger
* each coordinate stores learned semantic information

For example:

$$
\text{doctor} \rightarrow \vec v_{doctor}
$$

$$
\text{woman} \rightarrow \vec v_{woman}
$$

$$
\text{man} \rightarrow \vec v_{man}
$$

These vectors exist inside a high-dimensional semantic space.

---

## Semantic Geometry

One of the remarkable properties of embeddings is that semantic relationships become geometric relationships.

For example:

$$
\vec v_{king} - \vec v_{man} + \vec v_{woman}
\approx
\vec v_{queen}
$$

This suggests that semantic concepts may correspond to directions inside vector space.

---

# Defining a Gender Direction

## Constructing the Gender Axis

To identify gender-related information, we define a gender direction:

$$
g = \vec v_{woman} - \vec v_{man}
$$

This vector points from "man" toward "woman".

Geometrically:

* positive direction → more feminine
* negative direction → more masculine

---

## Normalization

We normalize the vector:

$$
g \leftarrow \frac{g}{|g|}
$$

so that:

$$
|g| = 1
$$

This simplifies projection calculations.

> [!INFO]
> A normalized vector is called a unit vector. Unit vectors allow dot products to directly measure projection lengths.

---

# Measuring Bias with Dot Products

## Dot Product Review

For two vectors:

$$
a,b \in \mathbb R^n
$$

their dot product is:

$$
a \cdot b= \sum_{i=1}^{n} a_i b_i
$$

Geometrically:

$$
a \cdot b= |a||b|\cos\theta
$$

where:

* $\theta$ is the angle between the vectors

---

## Scalar Projection

Because:

$$
|g| = 1
$$

the dot product:

$$
\vec v \cdot g
$$

becomes:

$$
|v|\cos\theta
$$

This is exactly the scalar projection of $\vec v$ onto the gender axis.

---

## Interpretation

If:

$$
\vec v \cdot g > 0
$$

then the word leans toward the feminine direction.

If:

$$
\vec v \cdot g < 0
$$

then the word leans toward the masculine direction.

If:

$$
\vec v \cdot g = 0
$$

then the word has no component along the gender axis.

---

# Orthogonal Projection

## The Core Idea

Any vector can be decomposed into:

$$
\vec v = \vec v_{\parallel}
+
\vec v_{\perp}
$$

where:

* $\vec v_{\parallel}$ is parallel to $g$
* $\vec v_{\perp}$ is orthogonal to $g$

This is called orthogonal decomposition.

---

## Projection onto the Gender Axis

The projection of $\vec v$ onto $g$ is:

$$
\text{Proj}_g(\vec v) = (\vec v \cdot g)g
$$

This extracts the gender-related component of the word vector.

---

## Geometric Meaning

The projection vector:

$$
(\vec v \cdot g)g
$$

is:

* parallel to $g$
* the closest point to $\vec v$ on the gender line

Geometrically, this is equivalent to dropping a perpendicular from the point onto the axis.

---

# Debiasing by Removing the Projection

## Debiased Vector

To remove gender information:

$$
\vec v_{debiased} = \vec v - (\vec v \cdot g)g
$$

This subtracts the gender component from the original vector.

---

## Geometric Interpretation

The original vector contains:

* a gender component
* a non-gender component

Debiasing removes only the gender component.

The remaining vector lies entirely inside the orthogonal complement space.

---

# Proof of Orthogonality

## Showing the New Vector is Orthogonal

Define:

$$
\vec v' = \vec v - (\vec v \cdot g)g
$$

Now compute:

$$
\vec v' \cdot g
$$

Substitute:

$$
(\vec v - (\vec v \cdot g)g)\cdot g
$$

Distribute:
$$
\vec v \cdot g
(\vec v \cdot g)(g\cdot g)
$$

Since:

$$
g\cdot g = 1
$$

we obtain:

$$
\vec v \cdot g \vec v \cdot g
$$

Therefore:

$$
=0
$$

Thus:

$$
\vec v' \perp g
$$

The debiased vector is orthogonal to the gender direction.

