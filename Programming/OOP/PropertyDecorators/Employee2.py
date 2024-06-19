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
        pass

emp_1 = Employee('Corey', 'Schafer', 50000)

emp_2 = Employee('Test', 'Employee', 60000)

print(Employee.raise_amt)
print(emp_1.raise_amt)
print(emp_2.raise_amt)