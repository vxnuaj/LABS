## AdaMax!

$ u_{t} = \beta^{\infty}_{2}v_{t-1} + \left(1-\beta^{\infty}_{2}\right)|g_{t}|^{\infty}$

$u_t = \max\left(\beta_{2}\cdot{v}_{t-1}, |g_{t}|\right)$

$\theta_{t+1} = \theta_{t} - \eta\frac{m_t}{v_{t}}$