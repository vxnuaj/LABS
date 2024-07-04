class Animal:
    def __init__(self, name, age, friendliness, species):
        self.name = name
        self.age = age
        self.friendliness = friendliness
        self.species = species

class Dog(Animal):
    def __init__(self, name, age, friendliness, species):
        super().__init__(name, age, friendliness, species) 

    def likes_walks(self):
        return True

    def __str__(self):
        return f"Name: {self.name}\nAge: {self.age}\nFriendliness: {self.friendliness}\nSpecies: {self.species}"

class Husky(Dog):
    def __init__(self, name, age, friendliness, species):
        super().__init__(name, age, friendliness, species)

class Poodle(Dog):
    def __init__(self, name, age, friendliness, species):
        super().__init__(name, age, friendliness, species)

class GoldenRetriever(Dog):
    def __init__(self, name, age, friendliness, species):
        super().__init__(name, age, friendliness, species)

class GoldenHusky(Husky, GoldenRetriever, Dog):
    def __init__(self, name, age, friendliness, species):
        super().__init__(name, age, friendliness, species)

if __name__ == "__main__":
    Zeus = GoldenHusky('Zeus', 7, 10, 'Husky')
    print(Zeus)
