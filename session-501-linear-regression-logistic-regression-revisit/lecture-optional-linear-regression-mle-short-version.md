#  Linear Regression — A Probabilistic Perspective

## 1. From Deterministic Fitting to Probabilistic Modeling

In introductory treatments of linear regression, we typically begin with an optimization-centric view: find parameters that minimize the mean squared error.

In this lecture, we take a fundamentally different approach. **We assume a probabilistic model that describes how the data is generated.** This shift in perspective is subtle yet profound. Rather than asking, "What parameters fit the data best?" we instead ask, "What parameters make the observed data most probable?"

## 2. The Generative Assumption

We assume that the target variable $y$ is generated according to the following process:

$$y = w^\top x + \epsilon$$

where $w$ represents the parameter vector and $\epsilon$ denotes noise. The critical assumption is that the noise follows a Gaussian distribution:

$$\epsilon \sim \mathcal{N}(0, \sigma^2)$$

This implies that the conditional distribution of $y$ given $x$ is:

$$y \mid x \sim \mathcal{N}(w^\top x, \sigma^2)$$

## 3. The Likelihood Function

Given a dataset $\{(x_i, y_i)\}_{i=1}^N$, we assume the samples are independent and identically distributed. The likelihood function is therefore:

$$p(\{y_i\} \mid \{x_i\}, w) = \prod_{i=1}^N \mathcal{N}(y_i \mid w^\top x_i, \sigma^2)$$

This represents the probability of observing the entire dataset given the parameters $w$.

## 4. Log-Likelihood

Working with products is computationally inconvenient, so we take the logarithm to obtain the log-likelihood:

$$\log p(\{y_i\} \mid \{x_i\}, w) = \sum_{i=1}^N \log \mathcal{N}(y_i \mid w^\top x_i, \sigma^2)$$

Substituting the Gaussian probability density function:

$$\log \mathcal{N}(y_i \mid w^\top x_i, \sigma^2) = -\frac{1}{2\sigma^2}(y_i - w^\top x_i)^2 + \text{const}$$

Thus, the total log-likelihood becomes:

$$\log p(\mathcal{D} \mid w) = -\frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - w^\top x_i)^2 + \text{const}$$

## 5. Maximum Likelihood Estimation

We define the maximum likelihood estimation problem as:

$$\max_w \log p(\mathcal{D} \mid w)$$

Substituting the log-likelihood expression, this becomes:

$$\max_w \left( -\frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - w^\top x_i)^2 \right)$$

## 6. The Key Equivalence

Maximizing the log-likelihood is equivalent to minimizing the sum of squared residuals:

$$\min_w \sum_{i=1}^N (y_i - w^\top x_i)^2$$

Therefore, **maximum likelihood estimation is exactly equivalent to least squares under the assumption of Gaussian noise.**

## 7. Why This Matters

This equivalence reveals a deeper insight: the mean squared error is not an arbitrary choice dictated by convenience. Rather, it emerges naturally from a probabilistic assumption. Specifically, Gaussian noise implies a squared error loss function.

## 8. Matrix Formulation

We can express the model in matrix notation as:

$$y = Xw + \epsilon$$

The likelihood then becomes:

$$y \sim \mathcal{N}(Xw, \sigma^2 I)$$

Maximum likelihood estimation corresponds to finding the mean of this Gaussian distribution that best explains the observed data.

## 9. Closed-Form Solution

Taking the gradient of the log-likelihood with respect to $w$ and setting it to zero yields:

$$w^* = (X^\top X)^{-1} X^\top y$$

This is the classical ordinary least squares solution. From the probabilistic perspective, **this is the parameter vector that maximizes the likelihood of the observed data.**

## 10. Generalizations

This probabilistic perspective enables immediate generalization to more complex scenarios:

- If the noise is not Gaussian, the loss function changes accordingly
- If the variance depends on $x$, we obtain weighted least squares
- If we place a prior distribution on $w$, we arrive at Bayesian linear regression

## 11. Beyond Maximum Likelihood: Ridge Regression

If we introduce a Gaussian prior on the parameters:

$$w \sim \mathcal{N}(0, \lambda^{-1} I)$$

then maximizing the posterior distribution (maximum a posteriori estimation):

$$\max_w \log p(w \mid \mathcal{D})$$

becomes equivalent to:

$$\min_w \sum_{i=1}^N (y_i - w^\top x_i)^2 + \lambda \|w\|^2$$

This is **ridge regression**, also known as L2 regularization.

## 12. A Probabilistic Interpretation

We can now reinterpret linear regression not merely as curve fitting, but as **probability modeling**. The objective is to model the conditional distribution of the target variable given the input features.

## 13. Summary

The key steps in this derivation are:

1. Assume Gaussian noise in the data generating process
2. Write the likelihood function
3. Take the logarithm to obtain the log-likelihood
4. Maximize the log-likelihood with respect to the parameters

This leads to the fundamental conclusion: **maximizing the log-likelihood under Gaussian noise is equivalent to minimizing the sum of squared errors.**

## 14. Closing Insight

What appears to be a simple optimization problem,

$$\min \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

is in fact **maximum likelihood estimation under a Gaussian generative model.** This intimate connection between probability theory and optimization constitutes one of the central unifying themes in modern machine learning.