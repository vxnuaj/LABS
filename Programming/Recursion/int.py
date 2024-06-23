"""
Problem: Write a recursive function to find the maximum value in a list of integers.

Example: If the input list is [3, 1, 4, 2, 5], the function should return 5.

"""


def max_value(data:list):
    if len(data) == 0:
        raise ValueError
    if len(data) == 1:
        return data[0]

    max_rest = max_value(data[1:])

    return data[0] if data[0] > max_rest else max_rest

def fib(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib( n - 2 ) + fib( n - 1 )

