import sys

class Student:
    def __init__(self, name:str , student_id:str, courses:list):
        self.name = name
        self.student_id = student_id
        self.courses = courses

    def add_courses(self, courses):
        if isinstance(courses, list):
            self.courses.extend(courses)
        elif isinstance(courses, str):
            self.courses.append(courses)
        else:
            raise TypeError('Incorrect Input Type!')

    def remove_courses(self, courses):
        for i in courses:
            self.courses.remove(i)

class Course:
    def __init__(self, course_code:str, course_name:str, students: list):
        self.course_code = course_code
        self.course_name = course_name
        self.students = students

    def get_students(self):
        return self.students

    def add_student(self, students):
        if isinstance(students, list):
            self.students.extend(students)
        elif isinstance(students, Student):
            self.students.append(students)
        else:
            raise TypeError('Incorrect Input Type! Must be List or String')

    def remove_student(self, students):
        if isinstance(students, list):
            for i in students:
                self.students.remove(i)
        elif isinstance(students, Student):
            self.students.remove(students)

juan = Student('Juan', '419293', ['CS321n', 'PHI101'])
rocky = Student('Rocky', '419294', ['CS101', 'ECE402'])
paul = Student('Paul', '293932', ['RCK123', 'PHY902'])

cs321 = Course('123', 'CS321n', [juan, rocky])

cs321.add_student([paul])
cs321.remove_student([juan])

print(f"Course Code: {cs321.course_code}")
print(f"Course Name: {cs321.course_name}\n")
print("Students:")
for i in cs321.students:
    print(f"- {i.name}")
      




    


