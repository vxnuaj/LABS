### Implementing Superconvergence

https://arxiv.org/pdf/1708.07120

First, we find the learning rate range through the LR Range test (proposed by Leslie Smith)

The following runs are performed with a cycle size of 10.

_First run:_

<img src = ../images/sc0.png width = 400>

After the first run, it seems that the accuracy seems to stagnate between the 8th and 9th epoch during the first cycle, once reaching a learning rate of about ~.72

For the next run, we'll then try a maximum learning rate of .7

_Second Run:_

The .7 learning rate still seemed too high, causing the accuracy to stagnate.

The maxium learning rate will now be dropped to .65

<img src = ../images/sc2.png width = 400>

Yet agian, the model seems to falter. The maximum learning rate will now be dropped to .60

<img src = ../images/sc3.png width = 400>


<...>

Reflection.

Failed Implementation, introduced mini-batch to make things interesting and got too complex quick. I think I should stop implementing things without valid motivation (i.e., a problem). I think perhaps I should start implementing things on more complex datasets to see them work with reason.

Otherwise, doing so for practice and just making them work could make sense