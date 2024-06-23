'''
Write a function to calculate
the sum of the digits of a non-negative
integer n. 

For example, if n is 1234, the sum of 
its digits is 1 + 2 + 3 + 4 = 10.
'''

def digit_sum(n):
    if n < 10:
        return n
    num = n % 10
    n = n // 10

    return num + digit_sum(n)

n = 3333

print(digit_sum(n))
