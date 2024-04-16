class Student:
    def __init__(self, name, house):

        if not name:
            raise ValueError("Missing name!")
    
        self.name = name
        self.house = house
        
    def __str__(self):
        statement = f"{self.name} from {self.house}!"
        return statement
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        if not name:
            raise ValueError("Missing name!")
        self._name = name
    
    # A property allows for us to call an attribute in a given class to a desired specific parameter
    @property
    def house(self):  
        return self._house
    
    # A getter allows for us to set rules when an attribute in a given class is assigned a new value.

    @house.setter
    def house(self, house):
        if house not in ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]:
            raise ValueError("Invalid house!")
        self._house = house

def main():
    student = get_student()
    print(student)

def get_student():
    name = input("Name: ").title()
    house = input("House: ").title()
    return Student(name, house)

if __name__ == "__main__":
    main()