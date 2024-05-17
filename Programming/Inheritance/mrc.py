'''Objective:
To understand and work with the method resolution order (MRO) and the super() function in diamond inheritance scenarios.

Instructions:
Create a base class Base with a method hello that prints "Hello from Base".
Create an intermediate class A that inherits from Base and overrides the hello method to print "Hello from A".
Create an intermediate class B that inherits from Base and overrides the hello method to print "Hello from B".
Create a subclass C that inherits from both A and B. Implement a method hello in C that calls the hello method from A using the super() function, then calls the hello method from B using the super() function.
Note:
You are not allowed to modify the Base, A, and B classes. Your task is to create the C subclass and implement the hello method as described above, demonstrating the method resolution order (MRO) and the use of the super() function in diamond inheritance.'''


class Base:
    def hello(self):
        print('Hello from Base')

class A(Base):
    def hello(self):
        print('Hello from A')

class B(Base):
    def hello(self):
        print('Hello from B')

class C(A, B):
    def hello(self):
        super().hello()
        super(A, self).hello()

c = C()
c.hello()