import random

class Hat:
    def __init__(self):
        self.houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
        return
    
    def sort(self, name):
        random_house = random.choice(self.houses)
        print(f"{name} is in {random_house}")
        return


hat = Hat()
hat.sort("Harry") #Prints what house the student is in.