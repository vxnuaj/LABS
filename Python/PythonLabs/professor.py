import sys
import random


def main():
    n = get_level()
    problems(n)
    return


def get_level():
    while True:
        try:
            n = input("Level: ")
            if n.isdigit() == True:
                n = int(n)
                if n not in [1, 2, 3]:
                    continue
                elif n in [1, 2, 3]:
                    return n
            else:
                continue
        except EOFError:
            sys.exit()


def problems(n):
    score = 0
    attempts = 0

    if n == 1:
        for prob in range(10):
            X = random.randint(0, 9)
            Y = random.randint(0, 9)
            Z = X + Y
            eq = f"{X} + {Y} = "
            usr_ans = 0
            while usr_ans != Z:
                usr_ans = input(eq)
                if usr_ans.isdigit() == True:
                    usr_ans = int(usr_ans)
                    if usr_ans == Z:
                        score += 1
                    elif usr_ans != Z:
                        attempts += 1
                        print("EEE")
                        if attempts == 3:
                            print(f"{X} + {Y} = {Z}")
                            attempts = 0
                            break
                    if prob == 9:
                        print(f"Score: {score}")
                        sys.exit()
                else:
                    continue
    elif n == 2:
        for i in range(10):
            X = random.randint(10, 99)
            Y = random.randint(10, 99)
            Z = X + Y
            eq = f"{X} + {Y} = "
            usr_ans = 0
            while usr_ans != Z:
                usr_ans = input(eq)
                if usr_ans.isdigit() == True:
                    usr_ans = int(usr_ans)
                    if usr_ans == Z:
                        score += 1
                    elif usr_ans != Z:
                        attempts += 1
                        print("EEE")
                    if attempts == 3:
                        print(f"{X} + {Y} = {Z}")
                        attempts = 0
                        break
                    if prob == 9:
                        print(f"Score: {score}")
                        sys.exit()
                else:
                    continue
    elif n == 3:
        for i in range(10):
            X = random.randint(100, 999)
            Y = random.randint(100, 999)
            Z = X + Y
            eq = f"{X} + {Y} = "
            usr_ans = 0
            while usr_ans != Z:
                usr_ans = input(eq)
                if usr_ans.isdigit() == True:
                    usr_ans = int(usr_ans)
                    if usr_ans == Z:
                        score += 1
                    elif usr_ans != Z:
                        attempts += 1
                        print("EEE")
                        if attempts == 3:
                            print(f"{X} + {Y} = {Z}")
                            attempts = 0
                            break
                    if prob == 9:
                        print(f"Score: {score}")
                        sys.exit()
                else:
                    continue


main()
