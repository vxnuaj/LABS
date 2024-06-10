**Exponential Decay**

$\alpha = \alpha \cdot e^{-k \cdot t}$

where $t$ is the current epoch or iteration and $k$ is the decay rate (a hyperparameter).

Make sure not to make $k$ too high as a model might actually stop learning

**Halving**

$\alpha = \frac{\alpha}{2}$

where $\alpha$ is halved after a set number of epochs.

This *halving* can be replaced instead by a division by any number, depending on how you might want to schedule the learning rate.

**Inverse Decay**

Very similar to exponential decay with the difference lying in the way it's computed:

$\alpha = \frac{\alpha}{1 + k \cdot t}$

