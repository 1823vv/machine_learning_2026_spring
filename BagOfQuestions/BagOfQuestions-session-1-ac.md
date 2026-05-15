## Question: Multiple Linear Regression and Row-Vector Convention

A house-price model uses three features:

```text
size, number_of_rooms, distance_to_city_center
```

1. Write the multiple linear regression model in scalar form.
2. Write the same model in row-vector form:

   $$
   \hat{y}=xW+b
   $$

3. If there are $d=3$ features, what are the shapes of $x$, $W$, $b$, and $\hat{y}$ for one sample?
4. If we stack $n=100$ samples as rows in $X$, what is the shape of $X$?
5. What is the shape of $\hat{Y}=XW+b$?
6. Draw the matrix multiplication $XW$ and mark the inner dimensions that must match.
7. Explain why each weight corresponds to one feature.
8. If the weight for `distance_to_city_center` is negative, what might that mean?

## Question: Feature Interpretation

1. Why should we be careful when interpreting weights if features have very different scales?
2. Which later Session 2 idea helps with this issue?
