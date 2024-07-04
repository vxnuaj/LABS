class Employee:
    raise_amt = 1.04
    
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay
        
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)
       
    def __repr__(self):
        return "Employee({}, {}, {})".format(self.first, self.last, self.pay)
    
    __str__ = fullname

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self.pay + other
        elif isinstance(other, Employee):
            return self.pay + other.pay
        else:
            raise ValueError(f"`other` must be type int, float, or instance of the Employee class!")

    def __len__(self):
        return len(self.fullname())

emp_1 = Employee('Juan', 'Vera', 10) 
emp_2 = Employee('Yoyo', 'Yuan', 20)

print(emp_1.fullname())
print(len(emp_1))