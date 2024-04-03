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

