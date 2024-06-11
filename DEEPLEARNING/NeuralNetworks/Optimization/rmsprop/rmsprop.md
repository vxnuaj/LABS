RMSprop, serving a similar purpose as [[Gradient Descent with Momentum]], computes the [[Exponentially Weighted Average]]s of gradients, with the difference that within the computation the $\theta$ value is squared:

$V_{\theta} = \beta V_{\theta - 1} + (1 - \beta) d\theta_t^2$

Then during the weight update, the gradient of $d\theta$ is divided by $V_{\theta}$ as:

$\theta = \theta - \alpha (\frac{d\theta}{\sqrt{V_{\theta} + \epsilon}})$

with an added $\epsilon$ to avoid a division by $0$.

> _By the way, the bias correction, the division by $\frac{1}{1 - \beta}$, isn't needed_

Keep note that RMSprop is more sensitive to the choice in learning rates, where higher learning rates tend to cause more instability in the training process.

With RMSprop at a learning rate, $\alpha = .001$, after 250 Epochs:

loss: 0.5318369393464288
acc: 81.71833333333333%

Faster learning than BatchNN after 250 epochs:

loss: 0.6555189972059645
acc: 76.72833333333332%