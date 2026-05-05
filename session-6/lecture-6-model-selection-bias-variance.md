# Bias–Variance Tradeoff

![](img/graphicalillustration.png)


---

## 1. Goal

Even with correct evaluation, models can fail in different ways.

We need to understand **why errors happen**.

---

## 2. Two Sources of Error

Model error comes from two main sources:

* bias
* variance

---

## 3. High Bias (Underfitting)

![](img/classification.png)



A model with high bias is too simple.

It cannot capture the structure of the data.


## Symptoms

* high training error
* high validation error
* both are similar


## Interpretation

The model is **systematically wrong**.

It misses key patterns.


## Example

* linear model on nonlinear data
* shallow decision tree

---

## 4. High Variance (Overfitting)

![](img/under_over_justalright.png)

![](img/bias-variance-overfitting.jpg)

https://visualize-it.github.io/polynomial_regression/simulation.html


A model with high variance is too sensitive to data.

It learns noise in the training set.


## Symptoms

* low training error
* high validation error
* large gap between them


## Interpretation

The model is **unstable across datasets**.

Small changes in data lead to large changes in predictions.


## Example

* very deep decision tree
* high-degree polynomial regression

---

## 5. Bias–Variance Tradeoff

![](./img/mc.jpg)

Increasing model complexity:

* reduces bias
* increases variance

Decreasing model complexity:

* increases bias
* reduces variance


> There is a **sweet spot** between underfitting and overfitting.

---

## 6. Learning Curves

![](img/ZahidHasan.png)


Learning curves plot:

* training error
* validation error
* vs dataset size



### High Bias

* both errors high
* curves close together


### High Variance

* training error low
* validation error high
* large gap

---

## 7. How to choose a good model

![](img/steps.png)


![](img/Bias-vs.webp)

