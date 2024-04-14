import sys


def main():
    one, two, three, four = usr_input()
    validation = validate(one, two, three, four)
    print(validation)
    return


def usr_input():
    ip = input("IPv4 address: ").strip()
    try:
        one, two, three, four = ip.split(".")
        return int(one), int(two), int(three), int(four)
    except ValueError:
        sys.exit("Invalid input!")


def validate(one, two, three, four):
    if (
        one in range(256)
        and two in range(256)
        and three in range(256)
        and four in range(256)
    ):
        return "True"
    else:
        return "False"

if __name__ == "__main__":
    main()
