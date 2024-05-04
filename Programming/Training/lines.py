import sys
import os

def main():
    file = get_file()
    f = open_file(file)
    num_lines = line_count(f)
    print(num_lines)
    return

#Counting Lines
def line_count(f):
    num_lines = 0
    for line in f:
        if line.strip().startswith("#") or line.strip() == "":
            continue
        elif not line.strip().startswith("#") or line.strip() == "":
            num_lines += 1
    return num_lines
        
#Opening file
def open_file(file):
    try:
        f = open(f"{file}", "r")
        f = f.readlines()
    except FileNotFoundError:
        sys.exit("File does not exist")

    return f

#Getting file from cmd line argument
def get_file():
        if len(sys.argv) == 2 and sys.argv[1].endswith(".py"):
            file = sys.argv[1]
            return file
        elif len(sys.argv) == 2 and sys.argv[1].endswith(".py") == False:
            sys.exit("Not a Python file")
        elif len(sys.argv) < 2:
            sys.exit("Too few command-line arguments")
        elif len(sys.argv) > 2:
            sys.exit("Too many command-line agruments")
main()