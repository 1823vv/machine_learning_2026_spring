## Question: Compare Linear and Logistic Regression Code From Scratch

We have already implemented `class MyOwnLinearRegression` from scratch with NumPy. Now we want to adapt the same general training structure to binary logistic regression. In both cases, the model stores `self.weights`, stores `self.bias`, loops for several gradient descent iterations, computes predictions, computes gradients, and updates parameters.

However, logistic regression is for classification. After computing the linear score

$$
z=XW+\mathbf{1}b,
$$

we pass the score through the sigmoid function to obtain a probability:

$$
\hat y = \sigma(z)=\frac{1}{1+e^{-z}}.
$$

Fill in the `____YOUR_CODE_HERE__N_____` blanks in the logistic regression fragment below.

```python
linear_model = np.dot(____YOUR_CODE_HERE__1_____, ____YOUR_CODE_HERE__2_____) + self.bias
y_predicted = self._sigmoid(____YOUR_CODE_HERE__3_____)

dw = (1 / n_samples) * np.dot(____YOUR_CODE_HERE__4_____, (____YOUR_CODE_HERE__5_____ - y))
db = (1 / n_samples) * np.sum(____YOUR_CODE_HERE__6_____ - y)

self.weights = self.weights - self.lr * ____YOUR_CODE_HERE__7_____
self.bias = self.bias - self.lr * ____YOUR_CODE_HERE__8_____
```

Your answer goes here, after the `:`:

- `____YOUR_CODE_HERE__1_____`:
- `____YOUR_CODE_HERE__2_____`:
- `____YOUR_CODE_HERE__3_____`:
- `____YOUR_CODE_HERE__4_____`:
- `____YOUR_CODE_HERE__5_____`:
- `____YOUR_CODE_HERE__6_____`:
- `____YOUR_CODE_HERE__7_____`:
- `____YOUR_CODE_HERE__8_____`:

Then answer the following short questions:

1. Name two things that stay almost the same when moving from linear regression code to logistic regression code.
2. Name two things that must change when moving from linear regression to logistic regression.
3. Explain why logistic regression outputs a probability before returning a class label, and why the final `predict` function usually returns class labels.
