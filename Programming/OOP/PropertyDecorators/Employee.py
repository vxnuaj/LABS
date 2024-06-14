class Employee:

    def __init__(self, first, last):
        self.first = first.title()
        self.last = last.title()

    @property
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)
    
    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    @fullname.setter
    def fullname(self, fullname):
        return self._fullname
        
emp_1 = Employee('John', 'Smith')

emp_1.fullname = 'Corey Schafer'

print(emp_1.first)
print(emp_1.email)
print(emp_1.fullname)
    