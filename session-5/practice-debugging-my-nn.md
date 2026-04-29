# Python Debugging: `practice-my_nn_with_bugs.py`

## Introduction

When programming, encountering errors, unexpected results, or runtime exceptions is completely normal. In fact, writing code that works perfectly the first time is almost impossible, especially for complex projects like neural networks.

What separates proficient programmers from beginners is **the ability to systematically locate, understand, and fix problems**—a skill known as debugging.

This tutorial is focused on practical **Python debugging techniques** using the file `practice-my_nn_with_bugs.py` as an example. While the file involves neural network code, the goal here is **learning how to debug effectively**, rather than teaching machine learning itself.

By following this tutorial, you will learn how to:

* Observe program behavior step by step
* Inspect variables, arrays, and shapes
* Identify the root cause of errors
* Apply a structured workflow to debug code efficiently

---

## 1. Understanding the Program Before Debugging

Before diving into debugging, it is essential to understand the **structure of the program**. Without context, debugging can become confusing and unproductive.

In `practice-my_nn_with_bugs.py`, the main components include:

* **Data loading**: `load_mnist_from_csv()`
* **Layer classes**: Dense layer, ReLU activation
* **Forward pass function**
* **Loss computation**: softmax + cross-entropy
* **Training loop**

For example, the **forward pass** of the network is implemented as:

```python
def forward(network, X):
    activations = []
    input = X
    for layer in network:
        input = layer.forward(input)
        activations.append(input)
    return activations
```

This function passes the input `X` through each layer sequentially and stores intermediate outputs. Understanding the **data flow** is critical for identifying where problems may occur.

---

## 2. Print Debugging

The simplest and most widely used debugging method is **printing intermediate values**. Printing allows you to inspect:

* Variable values
* Array shapes
* The sequence of execution

For example, when loading the MNIST dataset:

```python
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Validation labels shape: {y_val.shape}")
```

Sample output:

```
Training data shape: (54000, 784)
Training labels shape: (54000,)
Validation data shape: (6000, 784)
Validation labels shape: (6000,)
```

This quickly confirms whether the dataset was loaded correctly. If shapes are off, the problem is immediately visible.

### Printing Through the Network

Print debugging is also useful for monitoring **intermediate outputs in the neural network**:

```python
def forward(network, X):
    activations = []
    input = X
    for i, layer in enumerate(network):
        input = layer.forward(input)
        print(f"Layer {i} output shape: {input.shape}")
        activations.append(input)
    return activations
```

Output example:

```
Layer 0 output shape: (54000, 64)
Layer 1 output shape: (54000, 64)
Layer 2 output shape: (54000, 32)
Layer 3 output shape: (54000, 32)
Layer 4 output shape: (54000, 10)
```

If an unexpected shape appears, you immediately know **which layer may be the source of the problem**.

---

## 3. Assertions

Assertions allow you to **enforce assumptions** about your program state. They are particularly useful in numerical computations, where mismatched shapes or invalid values can silently propagate bugs.

Syntax:

```python
assert condition, "error message"
```

Example: checking input dimensions in a Dense layer:

```python
def forward(self, input):
    assert input.shape[1] == self.weights.shape[0], \
        "Input dimension does not match weight dimension"
    self.input = input
    return np.dot(input, self.weights) + self.biases
```

If the condition fails, Python raises:

```
AssertionError: Input dimension does not match weight dimension
```

Advantages of assertions:

* Detect problems early
* Document assumptions clearly
* Prevent silent bugs

---

## 4. Python Breakpoints (`pdb` / `ipdb`)

Interactive debugging is one of the **most powerful ways to find problems**. Python provides the built-in `pdb` debugger, but `ipdb` is an enhanced version with **tab completion, syntax highlighting, and better variable inspection**.

### Installing `ipdb`

```bash
pip install ipdb
```

### Basic Usage

A breakpoint pauses the program so you can inspect its state:

```python
import ipdb; ipdb.set_trace()
```

### Example: Debugging `hello.py`

Suppose you have a file `hello.py`:

```python
# hello.py
import sys

def full_name(first_name, last_name):
    """Returns the full name in capitalized form"""
    breakpoint()  # pause execution here for debugging
    name = f"{first_name.capitalize()}{last_name.capitalize()}"
    return name

if __name__ == "__main__":
    first_name = sys.argv[1]
    last_name = sys.argv[2]
    print(full_name(first_name, last_name))
```

Run the program:

```bash
python hello.py john lennon
```

The program pauses at `breakpoint()`. You are now inside `ipdb`, and can inspect variables:

```bash
ipdb> args
# first_name = 'john'
# last_name = 'lennon'

ipdb> sys.argv
# ['hello.py', 'john', 'lennon']

ipdb> name
# NameError: name 'name' is not defined
```

Notice the `NameError` occurs because execution is paused **before the assignment to `name`**. Step to the next line:

```bash
ipdb> next
ipdb> name
# 'JohnLennon'
```

Now you can see the problem: the space between names is missing. Fix it:

```python
name = f"{first_name.capitalize()} {last_name.capitalize()}"
```

Other useful `ipdb` commands:

| Command      | Description                              |
| ------------ | ---------------------------------------- |
| `n`          | Execute the next line                    |
| `s`          | Step into a function                     |
| `c`          | Continue execution until next breakpoint |
| `p variable` | Print variable value                     |
| `ll`         | List the current code context            |
| `args`       | Display function arguments               |
| `q`          | Quit the debugger                        |

`ipdb` allows you to **inspect variables, step through code, and understand control flow interactively**. Unlike print statements, it does not require modifying your code to see intermediate states.

---

### Using Breakpoints in `practice-my_nn_with_bugs.py`

You can insert `ipdb` in critical locations, such as **inside the training loop**:

```python
def train(network, X, y):
    activations = forward(network, X)
    import ipdb; ipdb.set_trace()  # pause execution here
    logits = activations[-1]
    loss = compute_loss(logits, y)
```

While paused, you can:

* Inspect shapes: `p logits.shape`
* Inspect outputs: `p logits[:5]`
* Check labels: `p y[:5]`
* Step through the backpropagation: `n`

This provides a **real-time window into your neural network** and helps you catch subtle errors, such as shape mismatches or NaNs.

---

### Advantages of `ipdb` over Print Debugging

* You can pause **exactly at the point of interest**
* Step through code **line by line**
* Inspect any variable at any time
* Avoid cluttering your code with print statements
* Works well with functions, loops, and complex data structures

> Tip: Remove `ipdb` breakpoints before committing code to version control. Many teams enforce pre-commit hooks to prevent accidental commits of debug statements.

---

## 5. Shape Checking

Incorrect array shapes are the most common source of errors in numerical code. In `practice-my_nn_with_bugs.py`, expected shapes include:

```
Input: (N, 784)
Dense1 weights: (784, 64)
Hidden layer output: (N, 64)
Dense2 output: (N, 32)
Final logits: (N, 10)
```

Use both **prints** and **assertions**:

```python
print("Logits shape:", logits.shape)
assert logits.shape[1] == 10
```

Regularly checking shapes ensures matrix operations are valid and prevents silent bugs.

---

## 6. IDE Debugging (VSCode / PyCharm)

Modern IDEs provide **visual debugging tools** that complement `ipdb`:

* Place breakpoints with a mouse click
* Step through code visually
* Watch variable values update in real-time
* Inspect the call stack

Example workflow:

1. Set a breakpoint in `train()`
2. Run the debugger
3. Observe variables such as `logits`, `grad_output`, `weights`, and `biases`

Watching how these values change over iterations **helps you understand program behavior**.
