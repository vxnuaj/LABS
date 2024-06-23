"""
Problem: Write a recursive function to calculate the nth Fibonacci number.

Definition: The Fibonacci sequence is a series of numbers in which each number 
is the sum of the two preceding ones, usually starting with 0 and 1.


"""

def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fib(n-2) + fib(n-1)

print(fib(6))
