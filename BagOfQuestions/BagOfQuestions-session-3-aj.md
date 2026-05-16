## Question: One-Hot Labels for Multiclass Classification

A digit classifier has $K=10$ possible classes: digits $0$ through $9$. The model outputs logits

$$
z \in \mathbb{R}^{1 \times 10}
$$

and softmax probabilities

$$
\hat{y} \in \mathbb{R}^{1 \times 10}.
$$

1. Why does the output layer need 10 logits?
2. If the true digit is $3$, describe the one-hot target vector $y \in \mathbb{R}^{1 \times 10}$ in words.
3. Write the multiclass cross-entropy loss

   $$
   \ell = -\sum_{k=1}^{K} y_k\log\hat{y}_k.
   $$

4. Explain why the one-hot label makes the loss become $\ell=-\log\hat{y}_c$ for the correct class $c$.
5. If the correct class probability is $\hat{y}_c=0.8$, compute the loss approximately using $\log 0.8 \approx -0.223$.
6. Draw a diagram connecting logits, softmax probabilities, one-hot label, and cross-entropy loss.
