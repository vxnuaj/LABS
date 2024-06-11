Adam, *Adaptive moment estimation*, is an extended version of [[Gradient Descent]], designed to improve training speeds in deep neural networks and converge quickly, making use of [[Momentum]] and [[RMSprop]], combined.

To compute Adam,

1. Compute the velocity term (1st moment) using the algorithm, $V_{d\theta }= (\beta_1 V_{d\theta - 1}) + (1 - \beta_1) d\theta$

2. Implement a bias correction to the velocity term, $\frac{V_d\theta}{1 - \beta}$

3. Compute the moving average of the accumulated squared gradients (2nd moment) using the algorithm, $S_{d\theta} = (\beta_2 S_{d\theta - 1}) + (1 - \beta_2)d\theta^2$

4. Implement a bias correction term, $\frac{S_{d\theta}}{(1 - B_2)}$

5. Perform the weight update, $\theta = \theta - \alpha(\frac{V_{d\theta}}{\sqrt{S_dw + \epsilon}})$, with the small $\epsilon$ value to avoid division by $0$

Then, you have 3 hyperparameters to tune:

1. Learning rate: $\alpha$
2. Momentum Term: $B_1$, typically initialized to $.9$
3. RMSprop Term: $B_2$, authors of the paper recommend to initialize to $.99$