class Employee:

    def __init__(self, first, last):
        self.first = first.title()
        self.last = last.title()

    @property
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)
    
    @property
    def fullname(self):
        if self.first or self.last == None:
            raise ValueError("Both first and last names must be set!")
        return '{} {}'.format(self.first, self.last)
    
    @fullname.setter
    def fullname(self, fullname):
        fullname = fullname.strip()
        if isinstance(fullname, str):
            try:
                first, last = fullname.split(' ')
                self.first = first
                self.last = last
            except (ValueError, AttributeError):
                raise ValueError ('Invalid input! Fullname must be str "First Last"')
        else:
            raise ValueError("Fullname must be a string!")
        
    @fullname.deleter
    def fullname(self):
        self.first = None
        self.last = None
        print("Deleted employee name!")

emp_1 = Employee('John', 'Smith')

emp_1.fullname = 'Juan Vera'



print(emp_1.first)
print(emp_1.email)
print(emp_1.fullname)

del emp_1.fullname
    
print(emp_1.fullname)