'''Objective:
To understand and resolve method conflicts in multiple inheritance in Python.

1. Create a base class A with a method hello that prints "Hello from A".

2. Create two intermediate classes B and C, both inheriting from A. B should have a method hello that prints "Hello from B", and 
C should have a method hello that prints "Hello from C".

3. Create a subclass D that inherits from both B and C. Implement a method hello in D that prints "Hello from D". Make sure that D uses the hello method from B.

Note:
You are not allowed to modify the A, B, and C classes. Your task is to create the D subclass that resolves the method conflict and uses the hello method from B.'''

class A:
    def hello(self):
        print("Hello from A")

class B(A):
    def hello(self):
        print("Hello from B")


class C(A):
    def hello(self):
        print("Hello from C")

class D(B, C):
    def hello(self):
        B.hello(self)

d = D()

d.hello()