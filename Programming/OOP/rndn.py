class Employee:

    def __init__(self, first, last):
        self.first = first
        self.last = last

    @property
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)
    
    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    @fullname.setter
    def fullname(self, name):
        first, last = name.split(' ')
        self.first = first
        self.last = last

        
    
emp_1 = Employee('John', 'Smith')

emp_1.fullname = 'Corey Schafer'

print(emp_1.first)
print(emp_1.email) 
print(emp_1.fullname)


'''
Notes
- @property allows you to call class methods as attributes, i.e., `print(emp_1.email)`
- to set new attritbutes when using @property decorators, we need to introduce @setter decorators
    it allows us to set an @property method and set it like any attribute


'''