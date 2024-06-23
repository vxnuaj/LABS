'''

Practicing Recursion

'''

import array

# Factorial Recursion

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

# Reversing a String
def rev_str(s):
    if s == '':
        return ''
    reverse = s[-1]
    return reverse + rev_str(s[:-1])

# Sum of Digits
def dig_sum(n):
    if n < 10:
        return n
    num = n % 10 # Gets the digit at the final index based on the remainder
    n = n // 10 # Eliminates the digit at the final index based on the integer division
    return num + dig_sum(n)

# Power of X to the Y
def power(x, y):
    if y == 0:
        return 1
    return x * power(x, y - 1)

# Fibonacci Sequence
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n - 2) + fibonacci(n - 1)

# Count Digits
def count_digits(n):
    if n < 10:
        return 1
    return 1 + count_digits(n // 10)

# Sum of Array Elements
def sum_array(array):
    if len(array) == 0:
        return 0
    num = array[-1]
    array = array[:-1]
    return num + sum_array(array)

# Check Element in Array
def check_array(array, element):
    if len(array) == 0:
        return False
    num = array[-1]
    array = array[:-1]
    return num == element or check_array(array, element)

# Calculate Exponential
def exponential(x, n):
    if n == 0:
        return 1
    return (x ** n) / ( factorial(n )) + exponential(x, n-1)
    
def rev_array(arr):
    if len(arr) <= 1:
        return arr
    rev_arr = array.array('i', [])
    rev_arr.append(arr[-1])  
    return rev_arr + rev_array(arr[:-1    ])

if __name__ == "__main__":
    
    # Factorial Recursion
    n = 5
    print(f"Factorial of {n} is {factorial(n)}\n")
    
    # Reversing a String
    s = 'vxnuaj'
    print(f"The reverse of {s} is {rev_str(s)}\n")
    
    # Sum of digits
    n = 456
    print(f"The sum of digits in {n} is {dig_sum(n)}\n")
    
    # Power
    x, y = 8, 2
    print(f"The power of {x} raised to {y} is {power(x, y)}\n")
    
    # Fibonacci
    n = 5
    print(f"The fibonacci of {n} is {fibonacci(n)}\n")
    
    # Count Digits
    n = 123
    print(f"The length of {n} is {count_digits(n)}\n")
    
    # Sum Array
    arr = array.array('i', [1, 2, 3, 5])
    print(f"The sum of {arr}, is {sum_array(arr)}\n")

    
    # Check Element in Array
    arr = array.array('i', [2, 3, 4, 5])
    element = 1
    print(f"The value in {element} is in {arr}? {check_array(arr, element)}\n")
    
    #Exponential
    x, n = 1, 10
    print(f"The exponential of {x} to the {n} terms in the expansion series is {exponential(x, n)}\n")
    
    # Reverse Array
    arr = array.array('i', [1, 2, 5])
    print(f"The reverse of {arr} is {rev_array(arr)}\n")