## Linear Regression

Linear regresion is a statistical method used to model the line of best fit amongst existing datapoints, to predict a next point, at least as accurate as a linear model can.

The line of best fit is modeled using $y = mx + b$ (think 6th/7th grade algebra), where 
- $m$ is the slope
- $b$ is the y-intercept / bias term

Just like this,

<img src = "https://www.scribbr.co.uk/wp-content/uploads//2020/02/simple-linear-regression-in-r-graph-example.png" width = 300>


where the value $y$, is predicted from an input data point $x$

> _Some implementations of linear regression models may include predictions of house prices, stock prices, weather of tomorrow, etc. Things that might have a linear correlation are a good fit for linear regression.

A linear regression model doesn't have to only have one set of weights (m) and inputs (x). It can have multiple as follows:

$price = w_{area} * area + w_{age} * age + b$,

- where $area$ and $age$ of a house is used to calculate the $price$ of a house.


In research papers, authors often write out linear models in the above format but they have a bigass equation that spans multiple lines making it difficult to understand (horrible formatting lol).

In higher dimensionality data, this becomes tedious therefore you can just use linear algebra notation:

$\hat{y} = w_{1} * x_1 + w_{2} * x_2 + ... + w_{d} * x_d + b$

- where $d$ is the index of datapoints in your dataset.

or you can collect all the features into single vector $x$ and params into vector $w$, as:

$\hat{y} = w^Tx + b$ 


In a linear regression model, the error (also known as the residual) for a specific prediction is the vertical distance between the predicted value on a regression line / line-of-best-fit, and the real/true observed value on an x,y plane.


<img src = "https://community.cloudera.com/t5/image/serverpage/image-id/25068iFF075A5AEC3B8528/image-size/medium?v=v2&px=400" width = 300>


To find the residual, a regression model needs to implement a loss function. A typically used loss function is the **Mean Squared Error (MSE).**

$E = \frac{1}{n} \sum_{i = 0}^{n} (y_i - \hat{y})^2$

- $i$ is the index for each datapoint
- $n$ is the total # of datapoints
- $y_i$ is the real value for  data point
- $\hat{y}$ is the prediction of the linear regression model

which can also be expressed as

$E = \frac{1}{n} \sum_{i = 0}^{n} (y_i - (mx_i + b))^2$,

where 

- $i$ is the index for each datapoint
- $n$ is the total # of datapoints
- $y_i$ is the real value for $ith$ data point
- $x_i$ is the feature value (our datapoint to predict $\hat{y}$)

> _Essentially, you subtract the prediction from the true value and average it over the total, n, datapoints_

The value of this function, the residual $E$, is what we need to minimize to ultimately make a linear regression model more accurate.

To minimize the residual, we want to find a function that minimizes that error by iteratively tweating $m$ and $b$ until we get an optimum line.

For each datapoint $i$ in the linear regression, we want to find the value of $m$ and $b$ by taking the gradient of the residual with respect to $m$ and $b$ (similar to neural nets & backprop).

This looks like this:

<img src = "imagebacklog/derivMSE.png" width = 500>


> Left is $∂m$ | Right is $∂b$

Once calcualting the deriv of a specific param, $m$ or $b$, to modify the value, we can use the update rule:

$m = m - ⍺ * \frac{∂E}{∂m}$

$b = b - ⍺ * \frac{∂E}{∂b}$,

- where ⍺ is the specified learning rate

This update rule moves the decreases the value of the loss by moving a value away (subtracting) from the steepest descent / gradient ($\frac{∂E}{∂param}$)

Typically, the lower the learning rate, the more precise and better a result will be becasue it allows for a model to pay more attention to specific details.


<details>
<summary> Resources </summary>

- [Neural-Nine's Linear Regression from Scratch in Python](https://www.youtube.com/watch?v=VmbA0pi2cRQ)

- [D2l.ai Linear Neural Networks Pytorch Version](https://github.com/dsgiitr/d2l-pytorch/tree/master/Ch05_Linear_Neural_Networks)
</details>
