## Question: Logistic Regression as One Neuron, Neural Network as Many Neurons

Start from the Session 2 logistic regression model and reinterpret it as a neural network.

1. Explain why logistic regression can be viewed as a single neuron.
2. Draw a diagram of logistic regression as:

   ```text
   input features -> linear score -> sigmoid -> probability
   ```

3. Now draw a hidden layer with 4 neurons receiving the same input features in parallel.
4. In your drawing, explain what “each column of the weight matrix corresponds to one neuron” means.
5. Explain in words what changes when we move from one neuron to many neurons.
6. Explain in words what changes when we stack multiple layers.
7. Why does this “stacking transformations” idea make neural networks more expressive than logistic regression?

## Question: Story Development

Imagine a model for classifying movie reviews as positive or negative.

1. What might a first hidden layer detect from a vectorized text input?
2. What might a deeper hidden layer combine from those first features?
3. What should the output layer produce for binary sentiment classification?
