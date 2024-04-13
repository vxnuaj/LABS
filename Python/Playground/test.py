import numpy as np


def main():
    user_input = f()
    hello(user_input)
    goodbye()
    shutdown()

def f():
    user_input = input("ENTER YOUR NAME: ")
    return user_input

def hello(user_input):
    print(f"Hello {user_input}!")
    
def goodbye():
    while True:
        good_bye = input("SAY GOODBYE: ")
        if good_bye == "goodbye":
            return good_bye

def shutdown():
    print("see you tomorrow!")


if __name__=="__main__":
    main()
