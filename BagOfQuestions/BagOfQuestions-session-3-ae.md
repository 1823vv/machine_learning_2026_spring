## Question: Softmax by Hand

Consider logits for a three-class classifier:

$$
z = [2, 1, 0]
$$

1. Write the softmax formula.
2. Compute the exponentials approximately using $e^2 \approx 7.39$, $e^1 \approx 2.72$, and $e^0=1$. Compute the sum of exponentials.
3. Compute the three softmax probabilities approximately. Check that the probabilities sum to approximately 1.
4. Which class is predicted by argmax? Draw a bar chart of the three logits and another bar chart of the three softmax probabilities.

## Question: Negative Logits Are Fine

Consider logits:

$$
z = [-1, 0, 1]
$$

1. Why are negative logits allowed?
2. Which class will have the largest softmax probability?
3. Explain why softmax cares about relative scores, not whether every score is positive.
