"""
Write a recursive function to calculate the number of ways to climb
`n` stairs, where you can either climb 1 or 2 stairs at a time.

Example: If n = 4, there are 5 ways to climb the stairs: 1+1+1+1, 
1+1+2, 1+2+1, 2+1+1, and 2+2.

"""

def stairs(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 2
    return stairs(n - 1) + stairs(n-2)

print(stairs(4))


