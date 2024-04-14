import numpy as np

x = np.array(([1,2], [3,4]))

y = np.array (([4, 5], [5, 6]))

ex1 = np.dot(x, y)

ex2 = np.dot(y, x)

print(f"{ex1} \n")

print(f"{ex2} \n")