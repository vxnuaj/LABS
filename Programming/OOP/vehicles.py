'''

https://pynative.com/python-object-oriented-programming-oop-exercise/#h-oop-exercise-1-create-a-class-with-instance-attributes

#EXERCISE 1-3

class Vehicle:
    def __init__(self, name, max_speed, mileage):
        self.name = name
        self.max_speed = max_speed
        self.mileage = mileage

    def __str__(self):
        return f"name: {self.name} \nspeed: {self.max_speed} \nmileage: {self.mileage}"

class Bus(Vehicle):
    def __init__(self, name, max_speed, mileage):
        super().__init__(name, max_speed, mileage)



bus = Bus('schoolbus', 50, 200)

print(bus)


'''
'''

# EXERCISE 4

class Vehicle:
    def __init__(self, name, max_speed, mileage):
        self.name = name
        self.max_speed = max_speed
        self.mileage = mileage

    def seating_capacity(self, capacity):
        return f"The seating capacity of a {self.name} is {capacity} passengers"
    

class Bus(Vehicle):
    def seating_capacity(self, capacity = 50):
        return super().seating_capacity(capacity)

bus = Bus('bus', 20, 200)

print(bus.seating_capacity())'''

''''
# understanding class and instance vars and overshadowing

class Vehicle:

    color = "White"

    def __init__(self, name, max_speed, mileage):
        self.name = name
        self.max_speed = max_speed
        self.mileage = mileage

bus = Vehicle('School Volvo', 180, 12 )
car = Vehicle('Audi Q5', 240, 18)

car.color = 'red'

print(car.__class__.color) # showing overshadowed class var
print(car.color) # showing new instance var'''

'''

# Exercise 5

class Vehicle:

    color = 'White'

    def __init__(self, name, max_speed, mileage):
        self.name = name
        self.max_speed = max_speed
        self.mileage = mileage

class Bus(Vehicle):
    pass

class Car(Vehicle):
    pass

car = Car('Audi Q5', 240, 18)
bus = Bus('School Volvo', 180, 12)

print(f"Color: {bus.color}, Vehicle: {bus.name}, Speed: {bus.max_speed}, Mileage: {bus.mileage}")
print(f"Color: {car.color}, Vehicle: {car.name}, Speed: {car.max_speed}, Mileage: {car.mileage}")
'''

'''
#EXERCISE 7 
class Vehicle:
    def __init__(self, name, mileage, capacity):
        self.name = name
        self.mileage = mileage
        self.capacity = capacity

    def fare(self):
        return self.capacity * 100

class Bus(Vehicle):
    
    def fare(self):
        fare_price = super().fare()
        fare_price += fare_price * .10
        return fare_price

School_bus = Bus("School Volvo", 12, 50)
print("Total Bus fare is:", School_bus.fare())'''

'''#EXERCISE 8
class Vehicle:
    def __init__(self, name, mileage, capacity):
        self.name = name
        self.mileage = mileage
        self.capacity = capacity

class Bus(Vehicle):
    pass

School_bus = Bus("School Volvo", 12, 50)

print(isinstance(School_bus, Vehicle))'''



