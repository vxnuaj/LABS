import sys

class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def get_grade(self):
        return self.grade
    
class Course:
    def __init__(self, name, max_students):
        self.name= name
        self.max_students = max_students
        self.students = []
    
    def add_student(self, student):
        if len(self.students) < self.max_students:
            self.students.append(student)
            return True
        else:
            sys.exit("Too many students!")

    def get_average_grade(self):
        value = 0
        for student in self.students:
            value += student.get_grade()
        average = value / len(self.students)
        return average
    
if __name__ == "__main__":
    s1 = Student('juan', 18, 82)
    s2 = Student('bill', 19, 75)
    s3 = Student('jane', 19, 75)
    
    course = Course('CS31n', 2)

    course.add_student(s1)
    course.add_student(s2)
    course.add_student(s3)
    print(course.get_average_grade())