## Question: Alternative Activation Layers

In our own NumPy neural-network implementation, we can replace ReLU with other activation functions while keeping the same layer interface: `forward(self, input)` returns the layer output, and `backward(self, grad_output)` returns the gradient with respect to the layer input.

1. Write the sigmoid formula and its derivative in terms of the sigmoid output. Then write the Leaky ReLU formula and its derivative using negative-side slope $\alpha=0.01$.
2. Write your code implementation for `class Sigmoid(Layer)` and `class LeakyReLU(Layer)`. In both classes, the forward method should store `self.input = input` before returning the activation.
3. Draw ReLU and Leaky ReLU on the same axes. Use the drawing to explain the possible advantage of Leaky ReLU when many inputs are negative.
4. Explain why sigmoid outputs are useful as probabilities in binary classification but can be less convenient inside many hidden layers.
