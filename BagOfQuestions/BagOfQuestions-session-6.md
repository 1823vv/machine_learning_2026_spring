
## Question
Examine this code that generates and visualizes a bivariate normal distribution:
```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = [0, 0]
sigma = [[1, 0.5], 
         [0.5, 1]]
n_samples = 1000

# Generate samples
data = np.random.multivariate_normal(mu, sigma, n_samples)

# Create scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.axis('equal')
plt.title('Bivariate Normal Distribution')
plt.xlabel('x')
plt.ylabel('y')
```

a) What does the `sigma` matrix represent? Explain each element in the matrix.
b) How does the value 0.5 in the `sigma` matrix affect the plot? How would we call the relationship between x and y in the case of `sigma` set to 0.5, -0.5, 0, and how will the plot look like?



###

What's Bivariate Gaussian Distribution (or, Bivariate Normal Distribution)? Write down everything (including math, figures etc.) you know about it. 
