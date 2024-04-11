# Neural Networks and Deep Learning | Week 2

> _I didn't take notes on Week 1, primarily due to lack of complexity_

## Binary Classification

A type of classification task where the classification is binary, meaning there are 2 choices, of which a sample is classified into 1.

If there were two classes within our dataset, 

1. Horse | 0
2. Zebra | 1

and a neural network was fed the following image:

<img src = "https://www.treehugger.com/thmb/qFhPReYPPaVgTtHBOthYeMJVeZ0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-1043597638-49acd69677d7442588c1d8930d298a59.jpg" alt = "Zebra" width ="300"/>


The model would clasify it into one of two classes, hopefully it being the class represented by 1, the Zebra. (At least, it would if it was trained right lol.)

### RGB

Images, expressed through RGB, can be split into 3 different color channels corresponding to Red, Green, and Blue. 

Each corresponding color channel can be represented by matrices of same dimensions as the original image which holds pixel values between 0 to 255.

When these matrices are stacked on top of one another (in numpy, would use `np.vstack`), the full color image would be created.

<img src = "https://miro.medium.com/v2/resize:fit:1100/format:webp/1*8k6Yk6MhED2SxF2zLctG7g.png" width= "500">

### Feature Vectors & RGB

To input an image into a neural network, we typicall unroll the RGB pixel values into a singular feature vector.

For example, the color channel matrix for Red of an image with dimensionality, 3x3, would be unrolled into a feature vector of dimensitonality 9x1 each corresponding to values ranging from 0 to 255.

<img src = "imagebacklog/featvec.png" alttext = "Rudimentary btw" width = "250"/>

### Notation for the Course

- `(x,y)` is a specific training sample where,
    - `n(x)` is the total number of features per sample `x`
    - `x ∈ ℝ`<sup>`n(x)`</sub>
    - `y ∈ {0, 1}`, given binary classification
- `m` is total number of training samples in a training sample where,
    - `{(x(1), y(1)), (x(2), y(2)),... (x(m), y(m))}` is a training set
    - May be written as `m_train` or `m_test`, for train and test samples respectively
- `X` is the matrix that holds the number of training samples and it's features.
    - `m` is the number of columns, indicating number of total samples
    - `n(x)` is the total number rows indicating total number of features.

        <img src = "imagebacklog/Xmatrix.png" width = "350">

    - In other cases, we might see conventions of using the transpose of the X matrix here (as I used in ["NNMNIST from scratch"](https://github.com/vxnuaj/np.mnistnn)). 
    
        Not using the transpose, and rather the convention used in this course makes the implementation of neural nets easier. 

    - In python, to check out the size of this matrix, when we run `X.shape` the result would be `(n(x), m)`

- `Y` is vector which holds the total amount of training labels for each sample in a dataset
    - `Y = [y(1), y(2),...y(m)]` 
    - The `Y.shape` will yield a dimensionality `(1,m)`


- `w` is the weight matrix, which is `∈ ℝ`<sup>`n(x)`</sup>
    - There has to be one weight, `w` for each parameter `n(x)`, which is why it's an `∈ ℝ`<sup>`n`<sub>`x`</sub></sup>

- `b` is the bias vector / matrix .

- `z` is the value of the weighted sum, `w`<sup>`T`</sup>`x + b`
    - The transpose is optional, contextual depending on the nature of the dataset


## Logistic Regression

**Logistic regression** is a learning algorithm when the output label(s) `Y` in a supervised learnign algorithm are all zero or one.

> _As a reminder, **supervised learning** is a type of machine learning that employs a labeled dataset to "teach" the algorithm to make a correct classification amongst different class labels._

Say we have a picture of a cat,

<img src = "https://t4.ftcdn.net/jpg/00/97/58/97/360_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.jpg" width = "300"/> | cat!


and a dataset where label `1` represents the cat class and label `0` represents the class, "not a cat"


In logistic regression, given `x`, we want `ŷ = P(y=1 | x)`, meaning we want to know that is the probability (`y`) that variable `x` correspondes to the class `1` (cat).

>_Essentially, according to our model, what would be the probability that the above picture is a cat?_

Params are `w` (weight) and `b` (bias).

We can't use traditional linear regression,
> `ŷ = w`<sup>`T`</sup>`x + b`


as when performing logistic regression, we want `ŷ` to be between 0 and 1 for a probability. Using linear regression can give us a value greater than 1 or a value less than 0.

> _**Reminder**: Linear regression is a means to predict the next value given a current (tangential) slope (think 6th grade mathematics). It serves as a means for prediction a the relationship when variables are completely linear_

Instead of using a pure linear model, logistic regression involes taking the linear regression. `ŷ =` `w`<sup>`T`</sup>`x + b`, and applying a sigmoid function to it.

>_**Reminder**: The transpose is optional, contextual depending on the nature of the dataset_


SIGMOID: `σ(z) = (1) / (1+e`<sup>`-z`</sup>`)`

<img src = "https://docs-assets.developer.apple.com/published/b7e6be05d8/3394561@2x.png" width = "400" > sigmoid function!

- If z is very large, then `σ(z)` will be nearing 1
- If z is very small, `σ(z)` will be nearing 0

In the earlier case, 

> `ŷ = P(y=1 | x)`
>
>_Essentially, according to our model, what would be the probability that the above picture is a cat?_

where the class for a cat is `1` and the input image **is** the image of a cat, using logistic regression, the model's job is to learn the optimal parameters `w` and `b` to output a 1.0 (100%) probability that the image belongs to class `1` (cat)

> **NOTE** _Will be using katex styling from now, instead of `code` styling_

So essentially given a training set, $[(x^1, y^1),...,(x^m, y^m)]$, you want $\hat{y}≈ y^i$

#### Logistic Regression Cost Function

A common loss function typically used in linear regression is the squared error, $(y - \hat{y})^2$. In logistic regression we can't use this as it introduces an optimization problem. Where logistic regression isn't linear, the loss function will have local minima. Using the squared error doesn't allow for a network to optimize it's weights past those local minima to a global optima.

The loss function for logistic regression is typically, 

$ L(\hat{y}, y) - ylog{\hat{y}} + (1-y)log(1-\hat{y})$ | Called the cross-entropy loss

Here, if $y=1$, we get $-log(\hat{y})$. To minimize the loss, we want $\hat{y}$ to be as big as small as possible or nearing 1 given that 0 ≤ $\hat{y}$ ≤ 1 due to the sigmoid activation function.

Then, if $y=0$, we get $-log(1-\hat{y})$. To minimize the loss, we want $\hat{y}$ to be as small as possible or nearing 0 given that 0 ≤ $\hat{y}$ ≤ 1 due to the sigmoid activation function.

The the cost function can be defined as the average of the loss function over all datapoints $i$,

$\frac{1}{m} \sum_{i=1}^{m} - y^ilog{\hat{y}^i} + (1-y^i)log(1-\hat{y}^i)$

$\frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^i, y^i)$

The loss function is the loss of 1 parameter, whilst the cost function is the averaged loss of all parameters.

Ultimately, logistic regression can be seen as a mini-neural network. Logistic regression computed on a single datapoint can be considered a neuron of a neural network!

>**Note**: For the full math, check [here](/Artificial-Intelligence/Machine-Learning/Logistic-Regression/logregmath.md)

## Gradient Descent

Given the cost function:

$J(w,b) = \frac{1}{m} \sum_{i=1}^{m} - y^ilog{\hat{y}^i} + (1-y^i)log(1-\hat{y}^i)$

our goal is to minimize the value of this cost function by dynamicall adjusting params $w$ and $b$ through gradient descent.

This is what gradient descent looks like visualized:

<img src = "https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/national/gradient-descent-convex-function.png" width = "350" />

Here, the horizontal axes represents $w$ and $w$, the parameters, while $J(w,b)$ is the value of the cost function represented as the height.

Ultimately, you want to find optimal values $w$ and $b$ to find the global optima of this convex function, $J$.

> _Here, $J$ is the log loss function which is a convex function for Logistic regression. If the function wasn't convex, you'd risk getting params $w$ and $b$ at a local minima stalling the training process._

For gradient descent, you'd initialize $w$ and $b$ either as 0 (`np.zeroes`) or a random value (`np.random.rand`). Either or works well, you'll eventually arrive at the same optima. 

After iterations of gradient descent, based on a learning rate (typically denoted as alpha or ⍺), $w$ and $b$ begin to be optimized towards values that optimize the loss to a global optima.

This is done through the update rule: 

$θ - ⍺ * (\frac{∂J}{∂θ})$

where 

- $⍺$ is the learning rate
- $\frac{∂J}{∂θ}$ is the gradient of the loss w.r.t to parameter $θ$
    - This gradient ultimately denotes how steep the slope of the loss function is at the value which $θ$ represents. The larger the gradient is, the further away it is from the global optima.

This update rule moves the value of $θ$ in the oppostie direction of the gradient to ultimately find the global optima.


## Computation Graphs, Derivatives, and Gradient Descent

This is a computation graph:

<img src = "https://colah.github.io/posts/2015-08-Backprop/img/tree-backprop.png" width = 400>



Essentially, it's a means to sketch out mathematical operations of a model into different parts that feed into each other.

It allows us to easily visualize the processes of the gradient descent in an understandable manner. With it, we're able to see more clearly the impact that specific params have on the total value of the loss function.

Let's take this one as an example:

<img src = "imagebacklog/computationgraph.png" width = 600>

_ 

Say we want to find $\frac{∂J}{∂b}$ and we know that param $b$ is equal to 9 while loss, $J$, is equal to $27$.

Then, say we add $.002$ to parameter $b$ to get $9.002$ and the loss, $J$, increases by $.006$ to $27.006$

From that, we know that the $\frac{∂J}{∂b}$ is equal to 3. As param $b$ increases by $.002$, loss $J$ increases by $3$ times as much indicating that the gradient/loss/derivative of $J$ w.r.t. to $b$ is $3$.

Then say that the value of $a$ is equal to to $9$ and as we adjust $a$ upward by $.001$ to $9.001$, the value of $J$ increases by $.009$ from the original value $27$ to $27.009$. 

This indicates that the  $\frac{∂J}{∂a}$ is equal to $9$ while the $\frac{∂b}{∂a}$ is equal to to $3$ given that the $\frac{∂J}{∂b}$ was also equal to $3$.

This relationship can be easily calculated and shown through the chain rule for back propagation as such:

$\frac{∂J}{∂a} = (\frac{∂J}{∂b})(\frac{∂b}{∂a})$

$\frac{∂J}{∂a} = (3)(3)$

$\frac{∂J}{∂a} = 9$

Then, if we change $a$, we can change $b$, which ultimately can change the gradient of the loss, $J$, to a minimal value over time through the update rule, $θ = θ - ⍺ * \frac{∂J}{∂θ}$

Let's find the gradient of an earlier parameter, say $z$

Say we increase $z$ by $.001$ and,
- $a$ increases by $.001$
- $b$ increases by $.003$
- $J$ increases by $.009$

$\frac{∂a}{∂z}$ is equal to 1 as an increase in param $z$ is equivalent to a same increase param $a$

From this, we can tell that $\frac{∂J}{∂z}$ is equal to 9, just like $\frac{∂J}{∂a}$.

Again, this can be computed using the chain rule as:

$\frac{∂J}{∂z} = (\frac{∂J}{∂b})(\frac{∂b}{∂a})(\frac{∂a}{∂z})$

> *We can replace $(\frac{∂J}{∂b})(\frac{∂b}{∂a})$ with $\frac{∂J}{∂a}$ as we've calculated that earlier through $\frac{∂J}{∂a}=(\frac{∂J}{∂b})(\frac{∂b}{∂a})$*

$\frac{∂J}{∂z} = (\frac{∂J}{∂a})(\frac{∂a}{∂z})$

$\frac{∂J}{∂z} = (9)(1)$

$\frac{∂J}{∂z} = 9$

Then, if we change $z$ we can change $a$ which then changes $b$ which then ultimately changes $J$. Through the update rule, $θ = θ - ⍺ * \frac{∂J}{∂θ}$, we can eventually change the loss $J$ to a minimal value. 

This is what a model does during back propagation in order to increase its accuracy.

## Logistic Regression & Gradient Descent.

Say we implement a linear regression model and need to go through a training step. 

This involves computing the gradient of the loss w.r.t to the parameters

<img src = "imagebacklog/logregcompgraph.png" width = 500>

To ultimately compute the loss, $L$ w.r.t the parameters, $w_1, w_2$ and $b$, we need to take the gradient with respect to $a$ and $z$ prior.

To find $\frac{∂L}{∂w_1}$:

$\frac{∂L}{∂z} = (\frac{∂L}{∂a})(\frac{∂a}{∂z})$

$\frac{∂L}{∂w_1} = (\frac{∂L}{∂z})(\frac{∂z}{∂w_1})$

Similarly, for $\frac{∂L}{∂w_2}$:

$\frac{∂L}{∂z} = (\frac{∂L}{∂a})(\frac{∂a}{∂z})$

$\frac{∂L}{∂w_2} = (\frac{∂L}{∂z})(\frac{∂z}{∂w_2})$

And again for, for $\frac{∂L}{∂b}$:

$\frac{∂L}{∂z} = (\frac{∂L}{∂a})(\frac{∂a}{∂z})$

$\frac{∂L}{∂b} = (\frac{∂L}{∂z})(\frac{∂z}{∂b})$

Afterward, to update each parameter, we implement the update rule as:

$θ = θ - ⍺ * \frac{∂L}{∂θ}$

where $θ$ is a parameter.

If on a training set, where the total samples is denoted by $m$, you'd take the sum of  the loss over all samples $m$ and average it to find your average loss over the training set.

$J(w,b) = \frac{1}{m} \sum_{i=1}^{m}L(a^i, y^i)$

where
- $L$ is the loss for $ith$ sample
- $J$ is the cost over the entire training set.

## Vectorization

Vectorization is the process of converting all parameters $θ_i$ or inputs $x_i$ from seperate values into a singular vector or matrix $Θ$ / $X$

Turning all parameters into a singular vector/matrix $Θ$ / $X$ allows for GPUs/CPUs to leverage built in functions that compute at a faster rate.

Whenever possible, avoid using for-loops!! It slows computations down!

## Broadcasting 

Broadcasting is a feature in numpy that allows for a matrix or vector to be appended to match the dimensions of another matrix to perform a specific operation, $+, -, *,$ or $/$

Say I have the following matrices:

$A = \begin{pmatrix} 2, 4, 5 \\ 4, 1, 3 \end{pmatrix}$ of dimensions $(2, 3)$

$B = \begin{pmatrix} 2, 3, 4 \end{pmatrix}$ of dimenstions $(1, 3)$

if I multiply $A · B$ as `np.dot(A, B)`, the dimensions of $B$ will align to match the dimensions of $A$

$(1, 3) \rightarrow (2, 3)$

$\begin{pmatrix} 2, 4, 5 \\ 4, 1, 3 \end{pmatrix}$ ·$\begin{pmatrix} 2, 3, 4 \\ 2, 3, 4 \end{pmatrix}$