## Question: ReLU Layer Code and Drawing

In `code-my_nn.py`, ReLU is implemented as a layer.

1. Write the mathematical formula for ReLU.
2. Draw the ReLU function.
3. What does ReLU output for inputs `[-3, 0, 2]`?
4. Write the Python code for `ReLU.forward` using `np.maximum`.
5. In `ReLU.forward`, why do we store `self.input`?
6. In `ReLU.backward`, what does this line create?

   ```python
   relu_grad = self.input > 0
   ```

7. For `self.input = np.array([-3, 0, 2])`, what is `relu_grad`?
8. Explain why ReLU makes the neural network nonlinear.

## Question: What If We Remove ReLU?

1. Suppose a network has several Dense layers but no activation functions between them. What kind of function is the whole network equivalent to?
2. Why is this a problem if we want to learn complex image-classification boundaries?
