''' 

Thank you @ Corey Schafer 

'''

import datetime

class Employee:

    # These two are CLASS VARIABLES

    num_of_emps = 0
    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first.title()
        self.last = last.title()
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    def apply_raise(self):
        self.pay = int(self.pay + self.raise_amt)

    # Changes the class variable to `amount` ?

    @classmethod
    def set_raise_amt(cls, amount):
        cls.raise_amt = amount

    @classmethod
    def from_string(cls, emp_str:str):
        try:
            first, last, pay = emp_str.split("-")
            return cls(first, last, pay)
        except (ValueError, AttributeError):
            print("Invalid string!")

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True


emp_1 = Employee('Corey', 'Schafer', 50000)
emp_2 = Employee('Test', 'Employee', 60000)

my_date = datetime.date(2016, 8, 10)

print(Employee.is_workday(my_date))