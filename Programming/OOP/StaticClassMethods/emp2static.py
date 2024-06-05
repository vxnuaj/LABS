import time
import datetime

class Employee:

    num_of_emps = 0 # These are class variables and can be modified via class methods
    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

        Employee.num_of_emps += 1

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)

    @classmethod
    def from_string(cls, employee):
        first, last, pay = employee.split('-')
        return cls(first, last, pay)

    @classmethod   
    def set_raise_amt(cls, amount):
        cls.raise_amt = amount

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        else:
            return True 

emp_1 = Employee('Corey', 'Schafer', 50000)
emp_2 = Employee('Test', 'Employee', 60000)

my_date = datetime.date(2024, 6, 4) # return weekday as an integer. This day is '1'

print(Employee.is_workday(my_date))