## Question: Multiple Linear Regression and Row-Vector Convention

A house-price model uses three features:

```text
size, number_of_rooms, distance_to_city_center
```

For short, call those features $x_1$, $x_2$, and $x_3$. Under the row-vector convention, one input is $x \in \mathbb{R}^{1 \times 3}$ and the model can be written as $\hat{y}=xW+b$.

1. Write the multiple linear regression model in scalar form and in row-vector form.
2. For one sample with $d=3$ features, state the shapes of $x$, $W$, $b$, and $\hat{y}$.
3. If $n=100$ samples are stacked as rows in $X$, state the shapes of $X$ and $\hat{Y}=XW+\mathbf{1}b$.
4. Draw the matrix multiplication $XW$ and mark the inner dimensions that must match.
5. If the weight for `distance_to_city_center` is negative, explain what that might mean.
