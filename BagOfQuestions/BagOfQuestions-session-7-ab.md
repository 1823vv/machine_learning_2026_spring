## Question: Dropout as Training Many Sub-networks

During training, dropout randomly deactivates neurons.

1. Write the Bernoulli sampling formula for a dropout mask using dropout probability $p$.
2. Explain the meaning of $p$, $1-p$, mask value 0, and mask value 1.
3. Write the dropout forward-pass formula using activation $h$ and mask $m$.
4. Draw a neural network before dropout and one sampled sub-network after dropout. Cross out the dropped neurons.
5. Explain why dropout can be interpreted as training many shared-parameter sub-networks.
6. Explain how dropout reduces co-adaptation between neurons.
7. During inference, should dropout still randomly remove neurons? Explain why or why not.

## Question: Dropout and Generalization

A student says: “Dropout makes training harder, so it must always make the model worse.”

1. Explain what is correct in this statement.
2. Explain what is wrong in this statement.
3. Draw two possible curves for training accuracy and validation accuracy, one without dropout and one with dropout.
4. In your drawing, show a case where dropout has lower training accuracy but higher validation accuracy.
