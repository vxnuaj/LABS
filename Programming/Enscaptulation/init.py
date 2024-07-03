class Person:
    def __init__(self, name, age, gender):
        self.__name = name
        self.__age = age
        self.__gender = gender
    
    @property   
    def Name(self):
        return self.__name
    
    @Name.setter
    def Name(self, name):
        self.__name = name
        
        
p1 = Person("vxnuaj", 18, 'm')

print(p1.Name)