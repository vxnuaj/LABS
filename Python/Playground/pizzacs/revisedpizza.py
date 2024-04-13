import csv
from tabulate import tabulate
import sys



if len(sys.argv) == 2:
    if sys.argv[1].endswith(".csv"):
        try:
            with open(sys.argv[1]) as f:
                reader = csv.DictReader(f) #Maps each row from file f into differeing dictionaries
                print(tabulate(reader, headers = "keys", tablefmt="grid"))
        except FileNotFoundError:
            sys.exit("File not found!")
    elif len(sys.argv) > 2:
        print("Too many cmd line args")
    elif len(sys.argv) < 2:
        print("Too lil cmd line args")
    