### Xavier Initialization

- Acts as a means to mitigate the vanishing gradient problem, yet when introduced raw inputs of high values, it can fail as the $\sqrt{\frac{1}{n_{in}}}$ might not be enough to reduce the weighted sum, $z$ as as viable input into $\sigma(x)$ or $tanh(x)$ (see 'instance 1' in [weightinit.ipynb](weightinit.ipynb))