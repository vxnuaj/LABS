using np.random.default_rng(seed = 1 ) for reproducible initial weights across models.

vanilla rms prop after 250 epochss with leraning rate of .001 and beta of .9 and lambda of 10

- acc: 82.795%
- loss: 0.49646184029620405

rmsprop + l2 regularization after 250 epochs w learning rate of .001, beta of .9, and lambd of 10:

- loss: 0.5194015414295412
- acc: 82.96166666666667%

for this model, might need a learning rate schedule as the loss stopped improving in the latter epochs (same with acc)

momentum after 250 epochs w learning rate of .1 and beta of .9:

- loss: 0.6489145010537625
- acc: 77.09333333333333%

adam after 250 epochs with learning rate of .1, beta_1 of .9 and beta_2 of .99 (without using log sum exp softmax trick):

- loss: 0.3777914098809062
- acc: 86.52833333333334%

batchnn.py after 250 epochs with al earning rate of .1

- acc: 76.63666666666667%
- loss: 0.6570295234410887

For a more definite comparison, here are the plots after 250 epochs

**ADAM**:

<img src = images/adam1.png width = 500>

**BatchNN.py**

<img src = images/batchnn.png width = 500>

**MOMENTUM**:

<img src = images/momentum.png width = 500>


**RMSprop with L2 reg**:

<img src = images/rmsl2.png width = 500>

**Vanilla RMSprop**

<img src = images/rms.png width = 500>

After 994 epochs (get numerical overflow after that) of ADAM

- 91.59% acc
- .223849 loss

<img src = images/adam1k.png width = 500>
