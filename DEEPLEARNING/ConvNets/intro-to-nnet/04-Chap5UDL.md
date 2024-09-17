# Loss Functions

### 5.1

You want to maximize the likelihood / probability output of the model, $f(x, \phi)$, that given an $x$ we get it's correpsonding label $y$.

A model can be easily trained to maximize this likelihood by training it to predict the parameters for a probability distribution $\theta$, where $\theta = \set{\mu, \sigma^2}$ in the case of a univariate (distribution for a single number) normal distribution. It's goal would be to find the proper $\theta$ that maximizes the probability of the correct label for $x$, which is $y$, occuring for the given $x$.

The **maximum likelihood criterion**, then computes the total combined probability of all $P(y|x)$, it's goal to maximize it's value. A maximal value of $1$ would indicate that our model is able to accurately map the input $x$ to it's corresponding label $y$.

$\hat{\phi} = \argmax{\prod_i^I [P(y_i | x_i)]}$

This is under the assumptions that the data is independent and identically distaributed ($i. i. d$).



