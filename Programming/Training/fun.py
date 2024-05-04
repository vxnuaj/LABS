import numpy as np
import sys
from pyfiglet import Figlet
from colorama import Fore, Back, Style


def main():
    zeroah = zero_nine()
    tewety = ten_twenty()
    tezeroah = mat_mul(zeroah, tewety)
    outcome = win_loss(tezeroah)

    guess = user_input()

    print(f"The sum of the matrix multiplication is {np.sum(tezeroah)}")
    print(f"Your guess was {guess}")
    f = Figlet(font='slant')
    outcome = f.renderText(f'{outcome}')
    print(Fore.RED + f"{outcome}")


def user_input():
    try:
        guess = int(input("What's your guess? "))
        return guess
    except ValueError:
        sys.exit("Invalid guess! Must be a digit!")

def zero_nine():
    zeroah = np.random.randint(0,9,(3,3))
    return zeroah

def ten_twenty():
    tewety = np.random.randint(10,20, (3,3))
    return tewety

def mat_mul(zeroah, tewety):
    tezeroah = np.dot(tewety, zeroah)
    return tezeroah

def win_loss(tezeroah):
    if np.sum(tezeroah) > 1500:
        return "Unlucky!"
    else:
        return "Lucky!"

if __name__ == "__main__":
    main()

