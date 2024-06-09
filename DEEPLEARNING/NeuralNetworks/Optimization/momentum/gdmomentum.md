## Gradient Descent with Momentum

`GDM.py`

> Originally had issue with exploding gradients, so i built `GDMwL2.py`, 

> Pure Gradient Descent with Momentum
> Beta value of .9
> Learning rate of .1

After 250 Epochs:

loss: 0.6354913710942346
acc: 77.87833333333334%

`GDMwL2.py`

> Initially was implemented as a means of regularizing exploding gradients in the original `GDM.py`, but then realized i made a mistake within `GDM.py` and was not an algorithm issue.

> Gradient Descent with Momentum + L2 Regularization
> Beta value of .9
> Regularization term of 10 (`lambd`)
> Learning rate of .1

After 245 Epochs:

>_I thought might've gotten lucky on the weight initialization, but improvements in speed are reproducible. Also, gotta start using `np.random.seed()`_

loss: 0.5581363764041517
acc: 81.58999999999999%

(I used a beta value of .8 & .7 later on, which both yielded similar results to a $\beta$ of `.9` with a difference in acc and loss of less than `1e-2`)

Now with a beta value of `.99`, after 245 epochs:

loss: 0.701982314824206
acc: 75.31166666666667%

Compared to a plain ol' neural network, `BatchNN.py`

> Learning Rate of .1

After 250 Epochs:

loss: 0.6555189972059645
acc: 76.72833333333332%

### Takeaways:

> Momentum works very well and speeds up the learning rate, especially when you introduce a regualrization term (though was not needed tbf for this dataset / model), though this is likely very well dependent on the context of the model architecture and the dataset

> I should start using `np.random.seed()` for reproducible results.