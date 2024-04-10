# Linear Algebra for Deep Learning

### Transpose

The transpose of a matrix is essentially the flipping of rows of the matrix with its columns.

<details>
<summary> Example 1 </summary>

Given the matrix $X$

$X = \begin{pmatrix} 2, 3 \\ 3, 2 \\ 5, 1 \\ 4, 2 \end{pmatrix}$

the tranpose of it is:

$X^T = \begin{pmatrix} 2, 3, 5, 4 \\ 3, 2, 1, 2 \end{pmatrix} $

</details>
<details>
<summary> Example 2 </summary>

Given the matrix $Y$

$Y = \begin{pmatrix}1, 2, 3 \\ 4, 5, 6 \\ 7, 8, 9 \\ 10, 11, 12 \end{pmatrix}$

the tranpose is given as:

$Y^T = \begin{pmatrix} 1, 4, 7, 10 \\ 2, 5, 8, 11 \\ 3, 6, 9, 12\end{pmatrix}$

</details>

### Matrix Multiplication

The dot product of $X$ and $W$ is the equivalent of: $X^Tw$ or $W^Tx$, depending on the order you compute it in. 

$ W:\begin{pmatrix} 2, 3, 3 \\ 3, 2, 1 \end{pmatrix} · X: \begin{pmatrix} 3, 2 \\ 1, 2 \\ 4, 3 \end{pmatrix} = \begin{pmatrix} (2 * 3 + 3*1 + 3*4), (2 * 2 + 3 * 2 + 3 * 3) \\ (3* 3 + 2 * 1 + 1 * 4), (3 * 2 + 2 * 2 + 1 * 3) \end{pmatrix} = Z: \begin{pmatrix} 21, 19 \\ 15, 13 \end{pmatrix}$ 

Here, the values in the product matrix $Z$ is the sum of the element wise product of the rows of $W$ with the columns of $X$.

When implementing the dot product in deep learning  during the calculation for the weighted sum, you typically want to multiply $W$ by $X$ rather than the inverse.

Typically,

- $W$ has the size of $(n,m)$, where rows $n$ is the number of neurons and columns $m$ is weights totalling to the number of neurons or input features from a previous layer
- Input matrix $X$ has the size of $(samples, n)$, where columns $n$ is the total number of features per sample

Therefore, when you take the dot product of $W$ and $X$, inputting $W$ as the first factor will multiply each of the the rows in $W$ by each of the columns in $X$ and output the sum of it. This will give you an output equivalent to the number of output neurons in a given layer which is what you'd want within a layer in a deep neural network. 

So, when using `np.dot`, make sure to input weights, $W$ as the first parameter rather than $X$

`np.dot(W, X)`


<details>
<summary> Example 1 </summary>

Given matrices

$A = \begin{pmatrix} 1, 2, 3 \\ 4, 5, 6 \end{pmatrix}$
$B = \begin{pmatrix} 7, 8 \\ 9, 10 \\ 11, 12 \end{pmatrix}$

the dot product ($Z$) of $A$ and $B$ is:

$\begin{pmatrix} (1 * 7 + 2 * 9 + 3 * 11), (1 * 8 + 2 * 10 + 3 * 12) \\ (4 * 7 + 5 * 9 + 6 * 11), (4 * 8 + 5 * 10 + 6 * 12)\end{pmatrix}$

$\begin{pmatrix} (7 + 18 + 33), (8 + 20 + 36) \\ (28 + 45 + 66), (32 + 50 + 72) \end{pmatrix}$

$Z:\begin{pmatrix} (58), (64) \\ (139), (154) \end{pmatrix}$
</details>

### Dot Product

Tldr; It's the vector version of the matrix multiplication

<details> 
<summary> Example 1</summary>

Given matrices

$x = [2 , 3,  5]$

$w = [1, 4, 6]$

and computed the dot product as $\vec{x} • \vec{w}$, we'd do:

$\begin{pmatrix} 2 \\ 3 \\ 5 \end{pmatrix} • \begin{pmatrix} 1 & 4 & 6 \end{pmatrix}$ 

which will ultimately equal:

$(2 * 1) + (3 * 4) + (6 * 5) = 2 + 12 + 30 = 44$

</details>