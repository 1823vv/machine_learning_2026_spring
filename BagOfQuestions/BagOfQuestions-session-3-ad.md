## Question: Output Layers Depend on the Task

The last layer of a neural network should match the prediction task.

Create a table with these columns:

```text
Task | Output dimension | Output activation | Typical loss | Interpretation
```

Fill the table for:

1. regression with one continuous target,
2. binary classification,
3. multiclass classification with 10 classes.

Then answer:

4. Why is a linear output natural for regression?
5. Why is sigmoid natural for binary classification?
6. Why is softmax natural for multiclass classification?
7. Why should the output layer be designed together with the loss function?
8. Draw three mini-pipelines, one for each task type.

## Question: MNIST Output

For MNIST digit classification:

1. How many output neurons are needed?
2. What does each output neuron represent before softmax?
3. What does each output value represent after softmax?
