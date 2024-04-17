class Student:
    def __init__(self, name, house):

        if not name:
            raise ValueError("Missing name!")
    
        self.name = name
        self.house = house
        
    def __str__(self):
        statement = f"{self.name} from {self.house}!"
        return statement
    

def main():
    student = get_student()
    print(student)

def get_student():
    name = input("Name: ").title()
    house = input("House: ").title()
    return Student(name, house)

if __name__ == "__main__":
    main()