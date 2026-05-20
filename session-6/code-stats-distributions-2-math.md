# Bernoulli Distribution and Binomial Distribution

## Bernoulli Random Variable

A Bernoulli random variable models a single experiment with exactly two possible outcomes:

$$
X \in \{0,1\}
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
X \sim \operatorname{Bernoulli}(p)
$$

---

## Probability Mass Function

The probability mass function of a Bernoulli random variable is

$$
P(X=x)=p^x(1-p)^{1-x}
$$

for

$$
x \in \{0,1\}
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
\mathbb{E}[X]=1\cdot p + 0\cdot (1-p)
$$

Therefore,

$$
\mathbb{E}[X]=p
$$

This means the average outcome approaches the success probability.

---

## Variance of a Bernoulli Variable

The variance measures uncertainty.

Using

$$
\mathrm{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2
$$

and noting that

$$
X^2=X
$$

for binary variables,

we obtain

$$
\mathbb{E}[X^2]=\mathbb{E}[X]=p
$$

Thus,

$$
\mathrm{Var}(X)=p-p^2
$$

which simplifies to

$$
\mathrm{Var}(X)=p(1-p)
$$

> [!INFO]
> Variance is largest when
>
> $$ p=0.5 $$
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
X \sim \operatorname{Binomial}(n,p)
$$

---

## Construction from Bernoulli Variables

Let

$$
X_1,X_2,\dots,X_n
$$

be independent Bernoulli random variables:

$$
X_i \sim \operatorname{Bernoulli}(p)
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
\mathbb{E}[X]=\sum_{i=1}^{n}\mathbb{E}[X_i]
$$

Because each Bernoulli variable has mean

$$
p
$$

we obtain

$$
\mathbb{E}[X]=np
$$

---

## Variance of the Binomial Distribution

For independent random variables,

$$
\mathrm{Var}(X)=\sum_{i=1}^{n}\mathrm{Var}(X_i)
$$

Each Bernoulli variable has variance

$$
p(1-p)
$$

Thus,

$$
\mathrm{Var}(X)=np(1-p)
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
X \approx \mathcal{N}\big(np,\,np(1-p)\big)
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
> $$ np \ge 5 $$
>
> and
>
> $$ n(1-p)\ge 5 $$

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
\mathrm{Var}(\bar{X})=\frac{\sigma^2}{n}
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
\mathrm{Var}(\hat{\nabla L})=
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
1-p
$$

where

$$
p
$$

is the drop probability.

Each neuron mask variable is

$$
m_i \sim \operatorname{Bernoulli}(1-p)
$$

---

## Applying the Mask

If neuron activation is

$$
a_i
$$

the dropout output becomes

$$
\tilde a_i = a_i\frac{m_i}{1-p}
$$

Thus:

* when

  $$
  m_i=1
  $$

  the neuron remains active

* when

  $$
  m_i=0
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
X \sim \operatorname{Binomial}(n,1-p)
$$

The expected number of active neurons is

$$
\mathbb{E}[X]=n(1-p)
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
a \sim \operatorname{Bernoulli}(p)
$$

This introduces stochastic exploration into the learning process.

---

## Expected Action

The expected value of the action is

$$
\mathbb{E}[a]=p
$$

Thus the policy probability directly controls long-term action frequency.
