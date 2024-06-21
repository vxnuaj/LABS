'''def factorial(n:int):
    result = 1
    if n > 0:
        for i in range(1, n + 1):
            result = result * i
        return result'''
    
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(999))