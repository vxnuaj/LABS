class Student:

    def __init__(self, name, house):
        self.name = name
        self.house = house

    def __str__(self):
        statement = f"{self.name} from {self.house}!"
        return statement
    
    @property
    def house(self):
        return self._house
    
    @house.setter
    def house(self, house):
        if not house:
            raise ValueError("You forgot to input a house!")
        elif house not in ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']:
            raise ValueError("House does not exist!")
        self._house = house

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if not name:
            raise ValueError("Input a name!")
        self._name = name

    @classmethod
    def get(cls):
        name = input("Name: ").title()
        house = input("House: ").title()
        return cls(name, house)
    

def main():
    student = Student.get()
    print(student)


if __name__ == "__main__":
    main()

