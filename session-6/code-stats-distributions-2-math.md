# Bernoulli Distribution and Binomial Distribution

## Bernoulli Random Variable

A Bernoulli random variable models a single experiment with exactly two possible outcomes:

$$
X \in {0,1}
$$

where

$$
X=1
$$

represents success, and

$$
X=0
$$

represents failure.

The probability of success is defined as

$$
P(X=1)=p
$$

and the probability of failure is

$$
P(X=0)=1-p
$$

The Bernoulli distribution is written as

$$
X \sim \text{Bernoulli}(p)
$$

---

## Probability Mass Function

The probability mass function of a Bernoulli random variable is

$$
P(X=x)=p^x(1-p)^{1-x}
$$

for

$$
x \in {0,1}
$$

This compact formula correctly produces both probabilities:

### Case 1 — Success

If

$$
x=1
$$

then

$$
P(X=1)=p^1(1-p)^0=p
$$

### Case 2 — Failure

If

$$
x=0
$$

then

$$
P(X=0)=p^0(1-p)^1=1-p
$$

---

## Mean of a Bernoulli Variable

The expected value is

$$
E[X]=1\cdot p + 0\cdot (1-p)
$$

Therefore,

$$
E[X]=p
$$

This means the average outcome approaches the success probability.

---

## Variance of a Bernoulli Variable

The variance measures uncertainty.

Using

$$
Var(X)=E[X^2]-E[X]^2
$$

and noting that

$$
X^2=X
$$

for binary variables,

we obtain

$$
E[X^2]=E[X]=p
$$

Thus,

$$
Var(X)=p-p^2
$$

which simplifies to

$$
Var(X)=p(1-p)
$$

> [!INFO]
> Variance is largest when
>
> $$
> p=0.5
> $$
>
> because uncertainty is maximal when both outcomes are equally likely.

---

# Binomial Distribution

## Definition

A binomial random variable counts the number of successes in multiple independent Bernoulli trials.

Suppose:

* number of trials:

  $$
  n
  $$

* success probability:

  $$
  p
  $$

Then

$$
X \sim \text{Binomial}(n,p)
$$

---

## Construction from Bernoulli Variables

Let

$$
X_1,X_2,\dots,X_n
$$

be independent Bernoulli random variables:

$$
X_i \sim \text{Bernoulli}(p)
$$

The total number of successes is

$$
X=\sum_{i=1}^{n}X_i
$$

This sum follows a binomial distribution.

---

## Probability Mass Function

The probability of obtaining exactly

$$
k
$$

successes is

$$
P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}
$$

where

$$
\binom{n}{k}=\frac{n!}{k!(n-k)!}
$$

counts the number of ways to arrange the successes.

---

## Mean of the Binomial Distribution

Since

$$
X=\sum_{i=1}^{n}X_i
$$

we use linearity of expectation:

$$
E[X]=\sum_{i=1}^{n}E[X_i]
$$

Because each Bernoulli variable has mean

$$
p
$$

we obtain

$$
E[X]=np
$$

---

## Variance of the Binomial Distribution

For independent random variables,

$$
Var(X)=\sum_{i=1}^{n}Var(X_i)
$$

Each Bernoulli variable has variance

$$
p(1-p)
$$

Thus,

$$
Var(X)=np(1-p)
$$

and the standard deviation becomes

$$
\sigma=\sqrt{np(1-p)}
$$

---

# Effect of Parameters

## Effect of Probability $p$

The parameter

$$
p
$$

controls the center and skewness of the distribution.

### Small $p$

When

$$
p \ll 0.5
$$

most probability mass concentrates near zero successes.

### Large $p$

When

$$
p \gg 0.5
$$

most probability mass shifts toward large numbers of successes.

### Symmetric Case

When

$$
p=0.5
$$

the distribution becomes approximately symmetric.

---

## Effect of Number of Trials $n$

As

$$
n
$$

increases:

* the distribution becomes wider
* the number of possible outcomes increases
* the distribution gradually resembles a normal distribution

The mean scales as

$$
np
$$

while the standard deviation scales as

$$
\sqrt{np(1-p)}
$$

---

# Normal Approximation to the Binomial

## Gaussian Approximation

For sufficiently large

$$
n
$$

the binomial distribution can be approximated by a normal distribution:

$$
X \approx \mathcal{N}\big(np,np(1-p)\big)
$$

with

$$
\mu=np
$$

and

$$
\sigma=\sqrt{np(1-p)}
$$

---

## Normal Probability Density Function

The Gaussian density is

$$
f(x)=\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
$$

This approximation becomes increasingly accurate as

$$
n
$$

grows.

> [!INFO]
> A common rule is that the approximation works well when
>
> $$
> np \ge 5
> $$
>
> and
>
> $$
> n(1-p)\ge 5
> $$

---

# Central Limit Theorem

## Statement of the CLT

Let

$$
X_1,X_2,\dots,X_n
$$

be independent random variables with finite mean and variance.

The normalized sum

$$
\frac{
\sum_{i=1}^{n}X_i-n\mu
}{
\sigma\sqrt{n}
}
$$

converges toward a standard normal distribution:

$$
\mathcal{N}(0,1)
$$

as

$$
n \to \infty
$$

---

## Intuition

Even if the original variables are not Gaussian, averaging many independent samples produces an approximately Gaussian result.

This explains why sums of uniform variables gradually become bell-shaped.

---

## Sample Mean Form

For the sample average

$$
\bar{X}=\frac{1}{n}\sum_{i=1}^{n}X_i
$$

the CLT implies

$$
\bar{X}\approx
\mathcal{N}
\left(
\mu,
\frac{\sigma^2}{n}
\right)
$$

---

## Important Consequence

As

$$
n
$$

increases, the variance of the sample mean decreases:

$$
Var(\bar{X})=\frac{\sigma^2}{n}
$$

Therefore larger samples produce more stable estimates.

---

# CLT in Deep Learning

## Full Gradient

The exact gradient over an entire dataset is

$$
\nabla L
=
\frac{1}{n}
\sum_{i=1}^{n}g_i
$$

where

$$
g_i
$$

is the gradient contribution from sample

$$
i
$$

---

## Mini-Batch Gradient

Instead of computing the full gradient, deep learning uses a mini-batch estimate:

$$
\hat{\nabla L} =
\frac{1}{B}
\sum_{i=1}^{B}g_i
$$

where

$$
B
$$

is the batch size.

---

## Why Mini-Batches Work

By the Central Limit Theorem,

$$
\hat{\nabla L}
$$

is approximately normally distributed around the true gradient:

$$
\hat{\nabla L}
\approx
\mathcal{N}
\left(
\nabla L,
\frac{\sigma^2}{B}
\right)
$$

Thus:

* larger batches reduce variance
* smaller batches produce noisier updates
* stochastic gradients remain statistically reliable

> [!WARNING]
> Extremely small batch sizes can produce unstable optimization because gradient variance becomes too large.

---

# Batch Size and Gradient Variance

## Variance Reduction

Suppose individual gradient samples have variance

$$
\sigma^2
$$

Then averaging a batch of size

$$
B
$$

gives

$$
Var(\hat{\nabla L})=
\frac{\sigma^2}{B}
$$

Therefore:

* doubling batch size halves gradient variance
* larger batches produce smoother optimization trajectories

---

## Tradeoff

Large batches:

* lower variance
* stable gradients
* higher memory cost

Small batches:

* noisier gradients
* faster iterations
* possible regularization effects

---

# Dropout as a Bernoulli Process

## Dropout Mask

In dropout, each neuron is independently kept with probability

$$
1-r
$$

where

$$
r
$$

is the dropout rate.

Each neuron mask variable is

$$
M_i \sim \text{Bernoulli}(1-r)
$$

---

## Applying the Mask

If neuron activation is

$$
h_i
$$

the dropout output becomes

$$
\tilde{h}_i=M_i h_i
$$

Thus:

* when

  $$
  M_i=1
  $$

  the neuron remains active

* when

  $$
  M_i=0
  $$

  the neuron is removed

---

## Expected Number of Active Neurons

For

$$
n
$$

neurons, the number of active neurons follows

$$
X \sim \text{Binomial}(n,1-r)
$$

The expected number of active neurons is

$$
E[X]=n(1-r)
$$

---

# Bernoulli Processes in Reinforcement Learning

## Stochastic Policies

In reinforcement learning, policies often output probabilities.

For binary actions:

$$
P(a=1)=p
$$

and

$$
P(a=0)=1-p
$$

---

## Action Sampling

The chosen action is sampled from a Bernoulli distribution:

$$
a \sim \text{Bernoulli}(p)
$$

This introduces stochastic exploration into the learning process.

---

## Expected Action

The expected value of the action is

$$
E[a]=p
$$

Thus the policy probability directly controls long-term action frequency.

---

# Summary

## Bernoulli Distribution

Models a single binary event:

$$
X \sim \text{Bernoulli}(p)
$$

with

$$
E[X]=p
$$

and

$$
Var(X)=p(1-p)
$$

---

## Binomial Distribution

Counts successes across multiple Bernoulli trials:

$$
X \sim \text{Binomial}(n,p)
$$

with

$$
E[X]=np
$$

and

$$
Var(X)=np(1-p)
$$

---

## Central Limit Theorem

Averages of many independent variables approach a Gaussian distribution.

This explains:

* normal approximation of binomial distributions
* stability of mini-batch gradients
* statistical behavior of stochastic optimization

---

## Deep Learning Connections

These probabilistic ideas appear throughout machine learning:

* dropout uses Bernoulli sampling
* mini-batch SGD relies on CLT
* stochastic policies use Bernoulli action sampling
* gradient estimates become Gaussian through averaging
