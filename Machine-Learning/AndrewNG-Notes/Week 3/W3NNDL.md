## Neural Networks and Deep Learning | Week 3

### Computing a Neural Network's Output

<details> 
<summary>Notation</summary>


$z_{1}^1 = w_{1}^{[1]T}x_1^{[1]} + b_{1}^{[1]} $

Here, the superscript refers to the current layer and subscript refers to the specific neuron in a given layer.

Second neuron in the first layer:

$z_{2}^1 = w_{2}^{[1]T}x_1^{[1]} + b_{2}^{[1]} $

First neuron in the second layer:

$z_{1}^2 = w_{1}^{[2]T}x_1^{[2]} + b_{1}^{[2]} $

</details> <br>

A neural network is essentially a combination of logistic regression computations. A single logistic regression model can be equivalent to a single neuron of a neural network.

In a neural network you typically have,
- One input layer
- X hidden layers (where X can be any real number)
- One output layer

Each layer holds a set number of neurons in which you feed forward your input data from an input layer.

In a singular neuron of a neural network, you compute the following computations:

- $ z = w^Tx + b$
- $ a = σ(z) = \frac{1}{1 + e^{-z}}$

$a$, then ultimately represents the prediction of a specific neuron, $\hat{y}$. In a regular logistic regression model, this would've been it! 

But in a neural network, unless it's at the output layer, this output is sent forward to another neuron for for further computation.

Say we have 4 neurons in our first hidden layer and we're feeding them forward. 

These are the computations:

$w_{1}^{[1]T}x_1^{[1]} + b_{1}^{[1]} = z_{1}^1 \rightarrow \frac{1}{1 + e^{-z_1^{1}}} = σ(z_1^{1})= a_1^{1}$

$w_{2}^{[1]T}x_2^{[1]} + b_{2}^{[1]} = z_{2}^1 \rightarrow \frac{1}{1 + e^{-z_2^{1}}} = σ(z_2^{1})= a_2^{1}$

$w_{3}^{[1]T}x_3^{[1]} + b_{3}^{[1]} = z_{3}^1 \rightarrow \frac{1}{1 + e^{-z_3^{1}}} = σ(z_3^{1})= a_3^{1}$

$w_{4}^{[1]T}x_3^{[1]} + b_{4}^{[1]} = z_{4}^1 \rightarrow \frac{1}{1 + e^{-z_4^{1}}} = σ(z_4^{1})= a_4^{1}$

Rather than computing each single activation seperate from each other, you can vectorize the parameters and inputs into matrices to make equations simpler

```math
\begin{pmatrix} w_1^{[1]T} \\ w_2^{[1]T} \\ w_3^{[1]T} \\ w_4^{[1]T} \end{pmatrix} · \begin{pmatrix}x_1 \\ x_2 \\ x_3 \end{pmatrix} + \begin{pmatrix} b_1^1 \\ b_2^1 \\ b_3^1 \\ b_4^1 \end{pmatrix}
```

This will ultimately yield a matrix $Z^{[1]}$ whicih can then be sent through a sigmiod activation for a matrix output of $a^{[1]}$

### Vectorizing Accross Multiple Training Samples

Rather than feeding forward each training sample 1 by 1 under a for loop as:

````
for i in range(m):
    z1 = np.dot(W,x) + b
    a2 = sigmoid(z1)
    z2 = np.dot(w, a2) + b
    a2 = sigmoid(z2)
````

where $m$ is the total number of training samples,

You can easily vectorize your inputs and feed them forward at once. 

>_The shape of this matrix might look as, $(samples, features)$, where the number of rows is the total number of samples. Each row, holds the total number of features._ 
>
>_The total number of columns - column[0] (because these are the labels) equals the total number of features per sample in the training set._
>
>_Or it could be the inverse where this matrix takes the size of $(features, samples)$, where the number of columns is the total number of samples. Each column, there holds the total number of features._
>
>_Then, the total number of rows - row[0] (because these are the labels) equals the total number of features per sample in the training set._
>
> _This depends on the nature of your dataset and the conventions you'd wish to use_

For more details on the mathematics refer to [here](/Artificial-Intelligence/Machine-Learning/Neural-Networks/ForwardPropagation.md)

### Activation Functions

**Sigmoid**, is mathematically defined as:

$σ(z) = \frac{1}{1 + e^{-z}}$

with the derivative being:

$\frac{e^{-z}}{(1+e^-{z})^2}$, 

though calculating this for a computer can be computationally expensive so it can be simplifed to:

$ \frac{1}{1 + e^{-x}} ·(1-\frac{1}{1+e^{-x}})$

which can be expressed in python as:

```
a * (1-a)
```

It looks like:

<img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png" width = 300>

**Tanh**, is mathematically defined as:

$σ(z) = \frac{e^z - e^{-z}}{e^z 1 + e^{-z}}$

with the derivative being:

$1 - (tanh(z))^2$

It looks like:

<img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Hyperbolic_Tangent.svg/2560px-Hyperbolic_Tangent.svg.png" width = 350>

Tanh almost always works better than sigmoid, as you center your activations around 0 between -1 and 1. This centering makes learning for the next layer a little bit easier.

The exception is at the output layer. If true label, $y$, is between 0 and 1, then using sigmoid makes sense at the output layer as it outputs a value between 0 and 1. This is helpful during binary classification tasks.

The issue with both **sigmoid** and **tanh**, is that as the values begin nearing the lower or upper ends of each function(0 or 1 for **sigmoid** and -1 or 1 for **tanh**), the gradient of each activation function begins to near 0. 

This can slow down gradient descent and the overall training of the model

This is why we have **ReLU**

**ReLU**, is mathematically defined as:

$a = max(0,z)$, where the we get $0$ as the output if $z < 0$ or $1$ if $z > 1$

with the derivative being:

$ 0 if z < 0, nan if z = 0, 1 if > 0$

It looks like:

<img src ="https://www.researchgate.net/publication/333411007/figure/fig7/AS:766785846525952@1559827400204/ReLU-activation-function.png" width=10>

**Leaky ReLU**, is defined as:

$a = max(0.01z, z)

with the derivative being:

$ 0..01 if z < 0.01, nan if z = 0, 1 if > 0$

It looks like:

<img src = "https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-25_at_3.09.45_PM.png" width = 350>


**Here's how you might choose activation functions:**

If your output or prediction is is a binary classification where 0 and 1 are the only choices, it's best to use **sigmoid** as the activation function for the output layer, given that it outputs a value between $0$ and $1$. There, a threshold value can be used to ultimately classify each predictions as $0$ and $1$

**ReLU** is seemingly the default choice for an activation function for the hidden layers and is what most people use.

 A disadvantage is that the derivative is always $0$ when the input, $z$ is less than $0$. 
 
 A solution is **Leaky ReLU** which has a slight slope when $z$ is negative. The advantage here is that the gradient of the postive half of ReLU doesn't near to 0, unlike sigmoid, meaning a neural network will learn faster. 
 
 Despite the negative half of ReLU nearing to 0, most of the input units $z$ might not be under 0 so it shouldn't be a trouble. Though, you can use Leaky ReLU if this becomes an issue.

> _You'd consider using **Leaky ReLU** when your input values, $x$, or parameters, $w$ or $b$ are negative._

Essentially, never use sigmoid unless you're doing binary classification and need it for your output layer. Tanh is superior.

The "default" or most used one is ReLU. If your inputs to ReLU $z$ become negative due to negative input values $x$ or negative params $w$ or $b$, you'd consider using Leaky ReLU

### Why Non-Linear Activation Functions?

A non-linear function introduces non-linearity into the model which allows us to model an output that varies non-linearly in according to observed patterns.

Without non-linearity, a neural network, no matter the amount of layers, would just behave as a single-layer model since you'd just be using linear functions. There wouldn't be a need for complexity.

You might only want to use a linear function in the output neuron in regression tasks.