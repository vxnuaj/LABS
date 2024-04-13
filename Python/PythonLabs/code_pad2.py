import csv

character = input("What's your DBZ name? ")
species = input("Where's your species? ")

with open("names.csv", "a") as file:
    writer = csv.DictWriter(file, fieldnames = ["character", "species"])
    writer.writerow({"character": character, "species": species})
