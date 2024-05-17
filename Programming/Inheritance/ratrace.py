class Employee:

    raise_amt = 1.5

    def __init__(self, first, last, pay):
        self.first= first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)

class Developer(Employee):
    raise_amt = 1.10
    
    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay)
        self.prog_lang = prog_lang
        return

class Manager(Employee):
    raise_amt = 3

    def __init__(self, first, last, pay, employees = None):
        super().__init__(first, last, pay)
        self.employees = []
        if isinstance(employees, list):
            self.employees.extend(employees)
        elif isinstance(employees, str):
            self.employees.append(employees)

    def hire_employee(self, emp:str):
        if emp not in self.employees:
            self.employees.append(emp)
            print(f"hired {emp.first}!")
        else:
            print(f"{emp.first} is already in your division!")
        return

    def fire_employee(self, emp:str):
        if emp in self.employees:
            self.employees.remove(emp)
            print(f"kicked tf out of {emp.first}!")
        else:
            print(f"{emp.first} was already fired dimwit!")
        return
    
    def print_emps(self):
        for emp in self.employees:
            print(emp.fullname())
        return



corey = Developer('Corey', 'Schafer', 50000, 'CPP')
mr_robot = Developer('Figure 1', 'Robot', 60000, 'Python')
vxnuaj = Developer('Juan', 'Vera', 50000, 'CPP, Python, C')

man = Manager('michael', 'scott', '50000', [corey, mr_robot])

man.fire_employee(corey)

man.hire_employee(vxnuaj)

print("\nEmployees:")
man.print_emps()