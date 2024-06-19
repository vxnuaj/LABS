import datetime

class Employee:
    num_of_emps = 0
    raise_amt = 1.04

    def __init__(self, first, last,  pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    def apply_raise(self):
        self.pay = int(self.pay + self.raise_amt)

    @classmethod
    def set_raise_amt(cls, amount):
        cls.raise_amt = amount

    @classmethod
    def emp2str(cls, empstr:str):
        first, last, pay = empstr.split('-')
        return cls(first, last, pay)

    @staticmethod
    def isworkday(day):
        if day.weekday() in [4, 5]:
            return False
        return True


my_date = datetime.date(2024, 6, 19)

print(Employee.isworkday(my_date))