## Question: Softmax Calculation and Probability Interpretation

Consider a three-class neural network classifier that outputs a logit vector $z = [z_1, z_2, z_3] = [2, 1, 0]$ for a given input example.

1. Write the general mathematical formula for the softmax function used to calculate the predicted probability $p_k$ for a specific class $k$.
2. Using the approximations $e^2 \approx 7.39$, $e^1 \approx 2.72$, and $e^0 = 1.00$, compute the numeric value of the normalization denominator (the sum of the exponentials).
3. Compute the three final softmax probabilities $[p_1, p_2, p_3]$ rounded to two decimal places. Verify that your calculated probabilities sum exactly to $1.00$.
4. Determine which class index is selected by an `argmax` operation over the probabilities. Sketch two simple side-by-side bar charts: one displaying the raw logit scores $z$, and the other displaying the resulting probabilities $p$.
5. Suppose a different input example yields a shifted logit vector $z_{\mathrm{new}} = [102, 101, 100]$. State the resulting probability vector $p_{\mathrm{new}}$ without performing explicit exponentiation. Use this scenario to explain why the softmax function depends entirely on the relative differences between logit values rather than their absolute magnitudes.
