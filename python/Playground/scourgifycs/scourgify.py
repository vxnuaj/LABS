from tabulate import tabulate
import sys
import csv

def main():
    read_csv = open_csv()
    write_csv = open_new_csv()
    for rows in read_csv:
        name_row = rows['name'].split(',')
        name_row.append(rows['house'])
        new_csv = get_new_csv(write_csv, name_row)
    return

def open_new_csv():
    write_csv = open(sys.argv[2], 'w')
    fieldnames = ['first name', 'last name', 'house']
    write_csv = csv.DictWriter(write_csv, fieldnames = fieldnames)
    write_csv.writeheader()
    return write_csv

def get_new_csv(write_csv, name_row):
    new_csv = write_csv.writerow({'first name': name_row[1], 'last name': name_row[0], 'house': name_row[2]})
    return new_csv

def open_csv():
    try:
        if len(sys.argv) == 3 and sys.argv[1].endswith(".csv"):
            read_csv = open(sys.argv[1], "r")
            read_csv = csv.DictReader(read_csv)
            return read_csv
        elif len(sys.argv) > 3:
            sys.exit("Too many command-line arguments")
        elif len(sys.argv) < 3:
            sys.exit("Too few command-line arguments")
        elif len(sys.argv) == 3 and not sys.argv[1].endswith(".csv") or not sys.argv[1].endswith(".csv"):
            sys.exit("Not a CSV File")
    except FileNotFoundError:
        sys.exit(f"Could not read {sys.argv[1]}")

if __name__ == "__main__":
    main()