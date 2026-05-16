## Question: ReLU Layer Code and Drawing

In our own NumPy neural-network implementation, ReLU is implemented as a layer with a `forward` method and a `backward` method.

1. Write the mathematical formula for ReLU, draw the ReLU function, and compute its output for inputs `[-3, 0, 2]`.
2. Write the Python code for `ReLU.forward` using `np.maximum`, and explain why `self.input` is stored.
3. In `ReLU.backward`, explain what `relu_grad = self.input > 0` creates. For `self.input = np.array([-3, 0, 2])`, what is `relu_grad`?
4. Explain why ReLU makes the neural network nonlinear.

## Question: What If We Remove ReLU?

A network has several dense layers but no activation functions between them.

1. What kind of function is the whole network equivalent to?
2. Why is this a problem if we want to learn complex image-classification boundaries?
