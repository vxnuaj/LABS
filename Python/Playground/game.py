import random
import sys

def main():
    n = level()
    rand_num = random.randint(1, n)
    guess(rand_num, n)
    return

def guess(rand_num, n):
    while True:
        try:
            user_guess = input("Guess: ")
            if user_guess.isdigit() == True and user_guess.isdigit() > 0:
                user_guess = int(user_guess)
                if user_guess > n:
                    print("Too large!")
                elif user_guess < n:
                    print("Too small!")
                elif user_guess == n:
                    print("Just right!")
                    sys.exit()
            elif user_guess.isdigit() == False:
                continue
        except EOFError:
            print("Game finished!")
            break
    return

def level():
    while True:
        try:
            n = input("Level: ")
            if n.isdigit() == True:
                n = int(n)
                return n
            elif n.isdigit() == False:
                continue
        except EOFError:
            print("Game finished!")


main()