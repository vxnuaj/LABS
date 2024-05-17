'''Objective:
To practice method overriding in Python.

1. Create a base class Animal with a method speak that prints "Animal speaks".
2. Create a subclass Dog that inherits from Animal and overrides the speak method to print "Dog barks".
3. Create a subclass Cat that inherits from Animal and overrides the speak method to print "Cat meows".

Note:

You are not allowed to modify the Animal class. Your task is to create the Dog and Cat subclasses and override the speak method in each subclass.'''

class Animal:
    def __init__(self):
        return
    
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    def speak(self):
        print("Dog barks")

class Cat(Animal):
    def speak(self):
        print("Cat meows")

dog = Dog()
dog.speak()  # Output: "Dog barks"

cat = Cat()
cat.speak()  # Output: "Cat meows"
