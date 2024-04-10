## Linear Algebra

#### Transpose

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

#### Dot Product

The dot product of $\vec{x}$ and $\vec{w}$ is the equivalent of: $x^Tw$ or $w^Tx$

>_Whether you take the transpose of $\vec{x}$ or $\vec{w}$, the result is the same._

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
