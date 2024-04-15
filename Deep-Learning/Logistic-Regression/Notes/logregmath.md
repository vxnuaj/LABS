## The Mathematics of Logistic Regression

>_Full implementation [here](logreg2.py)_

Given that we're using
- Weight matrix $W$
- Input matrix $X$
- Bias vector $b$
- Target vector $Y$

> We'll be using matrices, not individual vals to make things simpler (economists, smh).

Linear Regression: $z = W^TX + b$, where $z$ is the prediction/output of the linear regression (commonly referred to as the weighted sum within neural networks).

Sigmoid: $σ(z) = \frac{1}{(1+e^{-z})}$, where $σ(z)$ is the activation output of the logistic regression. We'll use $a$ instead of $σ(z)$ for simplicity.

Cross-Entropy Loss (Loss, not cost): $L(Y,a)= - Y log(a) + (1-Y)log(1-a)$

Cross-Entropy Cost: $J(Y,a) = \frac{1}{n}(-Y log(a) + (1-Y)log(1-a))$

So essentially,

1. Input matrix $X$ into $z = W^TX + b$

2. ApplY Sigmoid Activation through, $a = σ(z) = \frac{1}{(1+e^{-z})}$

3. Calculate the Loss: $L(Y,a) = -(Y log(a) + (1-Y)log(1-a))$

4. Take the gradients with respect to weights $W$: $\frac{∂J}{∂W} = (\frac{∂J}{∂a})(\frac{∂a}{∂z})(\frac{∂z}{∂w})$

5. Take the gradients with respect to bias $b$: $\frac{∂J}{∂b} = (\frac{∂J}{∂a})(\frac{∂a}{∂b})$

6. Update weights $W$: $W = W - ⍺ * \frac{∂J}{∂W}$

7. Update bias $b$: $b = b - ⍺ * \frac{∂J}{∂b}$

8. Repeat for `range(len(n))`

9. Calculate the Cost: $J(Y,a) = -\frac{1}{n}(Y log(a) + (1-Y)log(1-a))$

10. Repeat for `range(epochs)`, until you've trained your model.

btw, gradients of loss (not cost) w.r.t param $W$ and $b$ is:

- $\frac{∂L}{∂W} = (a - y) * X$
- $\frac{∂L}{∂b} = (a - y)$
- $θ = θ - ⍺ * \frac{∂J}{∂θ}$