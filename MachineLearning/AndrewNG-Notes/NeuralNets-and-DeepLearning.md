## Neural Network Basics | Week 2

> _I didn't take notes on Week 1, primarily due to lack of complexity_

### Binary Classification

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

<img src = "https://github.com/vxnuaj/Training/blob/main/MachineLearning/AndrewNG-Notes/imagebacklog/featvec.png?raw=true" alttext = "Rudimentary btw" width = "250"/>

### Notation for the Course

- `(x,y)` is a specific training sample where,
    - `n(x)` is the total number of features per sample `x`
    - `x ∈ ℝ`<sup>`n`<sub>`x`</sub></sup>
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

- `Y` is matrix which holds the total number of training samples in a dataset
    - `Y = [y(1), y(2),...y(m)]` 
    - The `Y.shape` will yield a dimensionality `(1,m)`

    