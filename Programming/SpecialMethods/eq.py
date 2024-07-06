'''

The script will return 'Testing' if in the expression, the `Person` class is read first.
It will return 'Testing2' if in the expression, the `Village` class is read first.
Python reads from left to right.

'''


class Person:
    def __init__(self, name):
        self.name = name
       
    def __eq__(self, other):
        return f"Testing - Person" 
        
class Village:
    def __init__(self, *people):
        self.people = []
        self.new_people(people)
        
    def new_people(self, people):
        self.people.extend(people)  
        
    def __eq__(self, other):
        return f"Testing - Village" 
         
vxnuaj = Person('vxnuaj')
tommy = Person('tommy')

vil = Village(vxnuaj, tommy)

print(vxnuaj == vil)