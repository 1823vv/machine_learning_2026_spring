## Question: Parameter Counting with Regularization Layers

Consider a neural network designed for handwritten digit recognition. The input is a flattened image of size $28 \times 28 = 784$ pixels. The network contains two hidden layers—the first with 64 units and the second with 32 units—followed by an output layer with 10 units. Each dense layer contains both weights and biases.

1. Calculate the number of weights and biases for each of the three layers individually, and provide the total number of trainable parameters in this base network.
2. If dropout layers are inserted after each hidden layer, how many trainable parameters do they add to the network? Explain your reasoning.
3. If a Batch Normalization layer is inserted immediately after the first hidden layer (64 units), it learns one scale parameter ($\gamma$) and one shift parameter ($\beta$) for each unit. How many new trainable parameters are added to the network by this layer?