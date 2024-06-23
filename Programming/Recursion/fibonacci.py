def fibonacci(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    return fibonacci(x - 2) + fibonacci(x - 1)


for i in range(10):
    num = fibonacci(i)

    print(num)
