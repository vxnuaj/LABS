## Testing the speed of BatchNormalization on Varying Models

Here, I compare the effect that BatchNormalization has in terms of training speed accross a variety of models each with different optimization algorithms.

I was curious to figure out, *what are the limtis to how fast I can train a neural network?*

Each model was run for **500 epochs**, therefore each model was run for **5000** training steps given the mini-batch size.

`Mini_BatchNormNN.py` is Vanilla BatchNorm, on a neural network trained on FashionMNIST with 10 minibatches, each batch totaling to 6k samples each.

> _More details under the corresponding sections_

### `Mini_BatchNormNN.py`

**Learning Rate ($\alpha$):** $.1$

**Final Accuracy:** $98.77$%

<img src = 'images/Mini_BatchNormNN.png' width = 500>

> _Comments: Did not expect Batch Normalization to be this effective... prior when training using mini-batches, my training would struggle to even near the % or loss when training in full batches. This exceeded my expectations, but now I'm excited to train the same model with learning rate decay (RMSprop)._

**Learning Rate ($\alpha$)**: $.5$

**Final Accuracy:** $max(99.95)$, true final: $99.91666$%

<img src = 'images/Mini_BatchNormNN2.png' width = 500>

### `RMSMini_BatchNormNN.py`

**Learning Rate ($\alpha$):** $.01$
**Decay Rate ($\beta$):** $.99$

**Final Accuracy**: $100.0$ (or about 99.99999 if we wanna be exact.)
**Final Loss:** $0.0010182603008388811$

<img src = 'images/RMSMini_BatchNormNN.png' width = 500>

>*Comments: ts is crazy. I'm reached 99% accuracy at about ~120 epochs, i believe. More excited to apply Adam now. Next CIFAR-10 as well.*