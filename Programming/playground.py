import numpy as np
import matplotlib.pyplot as plt

arr1 = np.array([1, 2, 3, 4])
arr2 = np.arange(0, 10, step = 4) # Returns nums by taking a 'step' based on the step param.
arr3 = np.linspace(0, 10, num = 4) # Returns X amount of nums where X = num (the input nums parameter)

'''
The difference is that np.linspace returns the numbers based on the ideal set of nums we want to return while
np.arange returus returns the numbers based on the step size we want to take.
'''


print(f"arr1: {arr1}")
print(f"arr1 dtype: {arr1.dtype}\n")
print(f"arr2: {arr2}")
print(f"arr2 dtype: {arr2.dtype}\n")
print(f"arr3: {arr3}")
print(f"arr3dtype: {arr3.dtype}\n")

rng = np.random.default_rng(seed = 1)
arr4 = rng.normal(size = (3,3))
arr4_alt = rng.normal(size = (3,3))

#print(f'arr4: {arr4}')
#print(f'arr4 ndim: {arr4.ndim}\n') # arr4 dim tells us how many dimensions (depth) an ndarray has.
#print(f'flatten arr4: {arr4.flatten()}\n') # ndarray.flatten() always returns a copy of the array, arr4'''

'''
numpy.ravel() tries to return a view into the same array when possible but may sometimes return a copy.

if it returns a view into the same array, then editing the previous array can edit the raveled array.

This might be useful if you want to use less memory and be more efficient, but may end up breaking your code if
you edit the original array.

'''

rav_arr_alt = arr4_alt.ravel() 
arr4_alt[0] = 99

arr5 = np.array([7, 8, 9])
arr6 = np.array([[1, 2, 3],[ 4, 5, 6]])

'''print(arr6, '\n')
print(arr5, '\n')
print(np.vstack((arr6, arr5))) # stacks arrays with equal size of the 1st dimension vertically'''

arr5 = arr5.reshape(-1, 1)
arr6 = arr6.T

'''print(arr5, '\n')
print(arr6, '\n')
print(np.hstack((arr6, arr5))) # stacks arrays with equal size of the 0th dimenion horizontally'''

