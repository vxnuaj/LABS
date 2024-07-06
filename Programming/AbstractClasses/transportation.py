from abc import ABC, abstractmethod

class Vehicle(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def hello_test(self):
        return f"Hello"

class Car(Vehicle):

    def __init__(self):
        pass
   
    def go(self):
        print("You drive the car!")

    def hello_test(self):
        return f"Hello"
        
class Motorcycle(Vehicle):
    
    def go(self):
        print("You ride the motorcycle!")
 
vehicle = Vehicle()
car = Car()

print(car.hello_test())


'''

A user can't instantiate a class that uses even a singular @abstractmethod

A class that uses an @abstract method, can't have a child class inherit
the abstract method, it needs to be overriden within the child class.

'''