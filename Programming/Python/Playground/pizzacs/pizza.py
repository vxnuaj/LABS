from tabulate import tabulate
import sys

def main():
    csv = load_csv(sys.argv[1]) #importing 
    new_csv = [] #initialzing lists of list to then feed into tabulate
    for line in csv: #for each line in the loaded csv file | *the only reason we can do this is because we used readlines()
        if line.endswith("\n"): # if the given line of the loaded csv files ends with "\n":
            new_line = line.split("\n") # split the line into a sub-list, when the aforementioned condition is true
            new_line.remove("") # in the new sub-list, remove the final item: the blank space, ""
            for line in new_line: # within the new_line, check for each character, if the new_line contains ","
                new_line = line.split(",") # if the new_line contains the character "," at variable line (remember line, is a variable which is assigned per iteration of new_line per char), split the list into sub strings.
            new_csv.append(new_line) # append each new_line into the new_csv for tabulate
    print(tabulate(new_csv, tablefmt="grid")) #print the tabulate
    return

def load_csv(file):
    try:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".csv"):
            file = sys.argv[1]
            csv = open(f"{file}", "r")
            csv = csv.readlines()
            return csv
        elif len(sys.argv) == 2 and not sys.argv[1].endswith(".csv"):
            sys.exit("Not a CSV File")
        elif len(sys.argv) > 2:
            sys.exit("Too many command-line aguments")
        elif len(sys.argv) < 2:
            sys.exit("Too few command-line arguments")
    except FileNotFoundError:
        sys.exit("File does not exist")


if __name__ == "__main__":
    main()