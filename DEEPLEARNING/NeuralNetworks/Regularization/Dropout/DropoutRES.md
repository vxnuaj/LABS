## Results from Implementing Dropout

> 64 hidden neurons, for Fashion-MNIST.

>_See `DropoutNN.py` and `nn.py` ; Those 2 were being compared._

**`DropoutNN.py` (see `dropoutNN.pkl` for weights)**

**First run**

Training
- Weight Init: Xavier
- Learning Rate: .1
- 500 Epochs
- Loss: 0.7741361864374599
- Acc: 71.92833333333334

Testing
- Loss: 0.636377936247884
- Acc: 76.05

**`nn.py` (see `nn.pkl` for weights)**

**First run:**

Training
- Weight Init: Xavier
- Leasrning Rate: .1
- 500 Epochs
- Loss: 0.4952886754906913
- Acc: 82.235
- Surpassed Training Acc of `DropoutNN.py` at ~ 150 Epochs

**Testing**
- Loss: 0.5002520334290989
- Acc: 82.47

> **Takeaway(s)** 
> - As we implement Dropout layers in a model, a model takes more iterations to converge as you're eliminating a random set of neurons per iteration, thereby reducing the amount of learning steps some neurons are able to take during gradient descent.
> - Dropout might not be as effective on FashionMNIST, for it to be, I'd need to increase model size but I don't have enough compute lol. I'll move onto google colab soon.

