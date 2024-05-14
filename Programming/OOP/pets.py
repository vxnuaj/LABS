class Pet:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def show(self):
        print(f"I am {self.name.title()} and I am {self.age} old!")

    def speak(self):
        print("idk what to say lol")

class Fish(Pet):
    pass

class Cat(Pet):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color

    def speak(self):
        print("Meow")

    def show(self):
        print(f"I am {self.name.title()} and I am the color {self.color}... {self.age} years old!")

class Dog(Pet):
    def speak(self):
        print("Bark")

if __name__ == "__main__":
    cat = Cat('luna', 9, 'black n white')
    cat.show()