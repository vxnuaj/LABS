class Student:
    def __init__(self, name, house):
        self.name = name
        self.house = house

        if not name:
            raise ValueError("Missing name!")
        
    def __str__(self):
        statement = f"{self.name} from {self.house}!"
        return statement
    
    @property
    def house(self):
        return self._house
    
    @house.setter
    def house(self, house):
        if house not in ["Gryffindor", "Hufflepuff", "Ravneclaw", "Slytherin"]:
            raise ValueError("Invalid house!")
        self._house = house

def main():
    student = get_student()
    student.house = "n4 privet drive"
    print(student)

def get_student():
    name = input("Name: ").title()
    house = input("House: ").title()
    return Student(name, house)

if __name__ == "__main__":
    main()