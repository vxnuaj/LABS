'''
This program creates an instance of a class DBZ in order to either 
1. Identify the species of characters based on the current dictionary (of characters) defined by self.name
2. Add new characters to the dictionary initiated by calling an instance of the class DBZ with their corresponding species based on the user input.

(Initial inputs : A is add, I is identify).

Eventually, the intention would be to add functionality to further progress this program. Some ideas being a mini storyline for each character, whether it be user made (add) or pre-made (through the identify functionality).
'''

class DBZ:
    def __init__(self):
        self.name = {'Goku':'Saiyan', 'Vegeta':'Saiyan', 'Frieza':'Frost Demon', 'Cell': 'Bio-Android'}

def main():
    dbz = DBZ() # Initializing with self.name dict!
    name, species, mode = add_identify(dbz)
    output(mode, name, dbz)
    return

def add_identify(dbz):
    mode = input(f"ADD OR IDENTIFY CHARACTER (A OR I): ").capitalize()
    if mode == "A":
        name, species = add_characters(dbz)
        return name, species, mode
    elif mode == "I":
        name, species = id_species(dbz)
        return name, species, mode
    else:
      raise ValueError("Invalid Input!")

def add_characters(dbz): #ADD NEW CHARACTERS
    name = input(f"CHARACTER NAME: ").title()
    species = input(f"SPECIES: ").title()
    dbz.name[name] = species
    return name, species

def id_species(dbz):
    name = input(f"CHARACTER NAME: ").title()
    if name not in dbz.name:
        raise ValueError("Character does not exist!")
    else:
        species = dbz.name[name]
    return name, species

def add_status(name, dbz):
    print(f"You just added {name} who is a {dbz.name[name]}!")
    return

def id_status(name, dbz):
    print(f"{name} is a {dbz.name[name]}!")

def output(mode, name, dbz):
    if mode == "A":
        add_status(name, dbz)
    elif mode == "I":
        id_status(name, dbz)

if __name__ == "__main__":
    main()