'''

Create a Python class to manage student information for a school. The class should have the following features:

- Class variable passing_percentage to store the minimum percentage required to pass.

- Instance variables name, roll_number, marks, and grade to store student information.

- Instance method calculate_grade(self) to calculate the grade based on the marks and the passing percentage.

- Class method set_passing_percentage(cls, percentage) to set the passing percentage for all students.

- Instance method is_pass(self) to check if the student has passed based on the passing percentage.

'''

class Student:
    passing_percentage = 59.5

    def __init__(self, name, roll_number, marks: float, grade: int):
        self.name = name
        self.roll_number = roll_number
        self.marks = marks
        self.grade = grade

    def calculate_grade(self):
        if self.marks >= Student.passing_percentage:
            return "Passing!"
        elif self.marks < Student.passing_percentage:
            return "Failing!"
        
    def update_marks(self, new_marks):
        self.marks = new_marks

    def display_student_info(self):
        print(f"Student Name: {self.name}")
        print(f"Roll Number: {self.roll_number}")
        print(f"Marks: {self.marks}")
        print(f"Grade: {self.grade}")
        print(f"Status: {self.calculate_grade()}")

    @property
    def grade(self):
        return self._grade
    
    @grade.setter
    def grade(self, grade):
        if isinstance(grade, int) == False:
            raise TypeError("Grade must be type int")
        elif grade > 12 or grade < 1:
            raise ValueError("Grade must be between 1st and 12th grade!")
        self._grade = grade

    @property
    def marks(self):
        return self._marks
    
    @marks.setter
    def marks(self, marks):
        if isinstance(marks, (int, float))  == False:
            raise TypeError("Mark must be type float or int!")
        elif marks > 100 or marks < 0:
            raise ValueError("Marks must be between 0 - 100")
        self._marks = marks

    @classmethod
    def set_passing_percentage(cls, passing_percentage):
        cls.passing_percentage = passing_percentage


vxnuaj = Student('Juan', '419293', marks = 60, grade = 1)

vxnuaj.display_student_info()


