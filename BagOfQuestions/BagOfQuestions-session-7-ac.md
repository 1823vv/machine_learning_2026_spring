## Question: Inverted Dropout Formula and Expectation

In inverted dropout, we scale the kept activations during training.

1. Write the formula for the inverted dropout mask:

   $$
   m_i \sim ?
   $$

2. If dropout probability is $p$, what is the probability that a unit is kept?
3. If a unit is kept, what value does the inverted dropout mask take?
4. If a unit is dropped, what value does the mask take?
5. Show that

   $$
   \mathbb{E}[m_i] = 1
   $$

   for inverted dropout.
6. Therefore, show that for an activation $h_i$,

   $$
   \mathbb{E}[h_i m_i] = h_i.
   $$

7. Explain why this makes inference simpler.

## Question: Original Dropout versus Inverted Dropout

Create a table comparing original dropout and inverted dropout with these rows:

1. train-time mask,
2. train-time scaling,
3. inference-time scaling,
4. whether inference is a no-op,
5. why the course implementation prefers inverted dropout.
