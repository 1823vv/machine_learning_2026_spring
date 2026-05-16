## Question: Alternative Activation Layers

A neural-network implementation can replace ReLU with other activation functions.

1. Write the formula for the sigmoid activation function.
2. Write a Python class `Sigmoid(Layer)` with `forward` and `backward` methods. Write the formula for Leaky ReLU.
3. Write a Python class `LeakyReLU(Layer)` with `forward` and `backward` methods. Draw ReLU and Leaky ReLU on the same axes.
4. Explain one practical advantage of Leaky ReLU compared with ReLU. Explain one practical disadvantage of sigmoid in deeper neural networks.

## Question: Activation Choice Scenario

A student trains a deep network with sigmoid activations and sees very slow learning.

1. Give one possible explanation related to saturation.
2. Why might ReLU or Leaky ReLU help?
3. Would changing the activation function change the number of trainable parameters? Explain.
