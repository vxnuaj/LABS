'''

Write a function to calculate the value of a base a raised to the power 
of an exponent b using recursion. The function should return a raised to 
the power of b.

'''

def cal_pow(a, b):
    if b == 0:
        return 1
    return a * cal_pow(a, b-1)

print(cal_pow(5, 5))


