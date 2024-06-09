RMSprop, serving a similar purpose as [[Gradient Descent with Momentum]], computes the [[Exponentially Weighted Average]]s of gradients, with the difference that within the computation the $\theta$ value is squared:

$V_{\theta} = \beta V_{\theta - 1} + (1 - \beta) d\theta_t^2$

Then during the weight update, the gradient of $d\theta$ is divided by $V_{\theta}$ as:

$\theta = \theta - \alpha (\frac{d\theta}{\sqrt{V_{\theta} + \epsilon}})$

with an added $\epsilon$ to avoid a division by $0$.

> _By the way, the bias correction, the division by $\frac{1}{1 - \beta}$, isn't needed_