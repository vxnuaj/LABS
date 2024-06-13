# Adam vs AdaMax

> *Reproducible results using np.random.default_rng(seed = 1)*

2 neural networks of hidden layer size 32,  with output layer of 10, for the FashionMNIST was run for 
1000 epochs (no mini batches)

One of them made use of Adam:

- Alpha: .1
- Beta-1 .99
- Beta-2: .9

<img src = adam/adam.png width = 400>

- Max Acc: 92.4866666666%
- Min Loss: 0.20832735167237934

The second made use of adamax:

- Alpha: .01
- Alpha when acc > 87%: .005
- Beta_1: .9
- Beta_1 when acc > 87%: .99
- Beta_2: .99

>_the learnign rate was scheduled as it was seen that performance began to oscillate after about 87%_

<img src = adamax/adammax.png width = 400>

- Max Acc: 92.985%
- Min Loss: 0.20146208389227274

Seems that adamax was able to reach a higher level of accuracy

>_Note that we tuned the hyperparams of adamax, while not tuning adam. tuning hyperparams of adam (i.e., scheduling learning rate) might've led to better results._