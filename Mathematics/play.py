import numpy as np
from scipy.linalg import lu_factor, lu_solve

A = np.array([[1,2],[3,4]])
print(A,'\n')

lu, piv = lu_factor(A)
b = np.eye(A.shape[0])
rref = lu_solve((lu, piv), b)

print(rref)

