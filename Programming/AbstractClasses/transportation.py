from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def __init__(self):
        pass

class Car(Vehicle):
   
    def go(self):
        print("You drive the car!")
        
class Motorcycle(Vehicle):
    
    def go(self):
        print("You ride the motorcycle!")
 
vehicle = Vehicle()
car = Car()
motorcycle = Motorcycle()

car.go()
motorcycle.go()


'''

Say we had a user wanting to create a class for a vehicle.

We wouldn't want them to use the Vehicle class, because it isn't implemented well enough for that task

We'd want a user to use the Casr and MOtorcycle class isntead.

To prevent users from using the vehicle class, we can use abstract classes. 


'''