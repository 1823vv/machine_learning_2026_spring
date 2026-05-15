
## Question 

Fill in the blank in the following text:

In our Neural Network from scratch implementation (as well as for the underlying implementation of PyTorch and TensorFlow), for Dropout, we have 
```python
self.mask = np.random.binomial(1, 1 - p, size=input.shape) / _____YOUR_TEXT_HERE_1_____
``` 

This is called **Inverted Dropout**, with the purpose of keeping the expected activation magnitude constant. When we apply Dropout during training, we randomly "drop" (zero out) each unit with probability $p$. **If we only did**

```python
mask = np.random.binomial(1, 1 - p, size=input.shape)
output = input * mask
```

**then** on average only a fraction `_____YOUR_TEXT_HERE_2_____` of our units **would** be "alive," so the total activation magnitude **would** shrink by a factor of `_____YOUR_TEXT_HERE_3_____`.  **To compensate**, we can **scale up** the remaining units by `_____YOUR_TEXT_HERE_4_____` **so that** the expected sum of activations stays the same as it would have been without dropout. 

By scaling at *training* time, we don't need to do any special scaling at *inference* time—when we set `self.training = _____YOUR_TEXT_HERE_5_____`, our `forward` simply returns the raw inputs (no mask), and we've already preserved the correct activation magnitudes.

By contrast, the "original" (or naïve) dropout implementation works like this:

1. **Training**  
   ```python
   mask = np.random.binomial(1, 1 - p, size=input.shape)
   output = input * mask
   ```
   – We drop units but **do not** rescale them.  
2. **Inference**  
   ```python
   output = input * _____YOUR_TEXT_HERE_6_____
   ```
   – We multiply every activation by the keep probability `_____YOUR_TEXT_HERE_7_____` to match the expected magnitude during training.

