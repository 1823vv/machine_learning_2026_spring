# Bag of Questions - Session 4
> For Session 4, we will focus on code `code-my_nn.py`, with some elements from lecture md files



## Question

What is the mathematical formula for cross-entropy loss? 


## Question


What is the difference between **logits** and **probabilities**?


Why is `np.max(logits, axis=1, keepdims=True)` subtracted before exponentiation in softmax? What's the mathmatical explanation for doing this?



## Question

In our NN from scratch implementation, If we replace ReLU with Sigmoid, what changes mathematically and practically?

Write down the python code for `Sigmoid` class. 

Your code should start with `class Sigmoid(Layer):`.

What about the code of `class LeakyReLU(Layer):`?


## Question

1. Point out what issues we have with the code here for the class Dense in our NN from scratch, and propose how to fix things.

```
class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(
            2.0 / input_units
        )
        self.biases = np.zeros(output_units)

    def forward(self, input):
        self.input = input
        return input * self.weights + self.biases

    def backward(self, grad_output):
        grad_weights = self.input.T * grad_output
        grad_biases = np.sum(grad_output, axis=0)
        grad_input = grad_output * self.weights.T
        self.weights = self.weights + self.learning_rate * grad_weights
        self.biases = self.biases + self.learning_rate * grad_biases
        return grad_input
```

2. Let's assume now you have the correct version of class Dense. What's the self.input in  `def forward`?  What's the input_units in  `def __init__`? What's the output_units in  `def __init__`? 


3. Draw a schema/figure for represention a NN of mnist recogonization with our NN from scratch implementation, featuring Input Layer, Dense Layers, Relu layers and output layer. The output layer should be decoupled into  two layers: logits layer and softmax activation output layer.

4. He initialization math:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

But in the code we don't see 0 in the `def __init__`, how do you explain things?



## Question

Fill the the `____YOUR_CODE_HERE__N_____`. 

```
def train(network, X, y):
    """Train the network on a batch of examples"""
    # Forward pass
    activations = ____YOUR_CODE_HERE__1_____(network, X)
    logits = activations____YOUR_CODE_HERE__2_____

    # Compute loss and initial gradient
    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)

    # Backward pass (backpropagation)
    grad_output = grad_logits
    for i in range(len(network))[____YOUR_CODE_HERE__3_____]:  # Reversed order
        layer = network[i]
        grad_output = layer.backward(____YOUR_CODE_HERE__4_____)

    return loss
```
Your answer goes here, after the `:`:
- `____YOUR_CODE_HERE__1_____`: 
- `____YOUR_CODE_HERE__2_____`: 
- `____YOUR_CODE_HERE__3_____`: 
- `____YOUR_CODE_HERE__4_____`: 


