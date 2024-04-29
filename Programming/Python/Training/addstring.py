class Group:
    def __init__(self, person1, person2):
        self.person1 = person1
        self.person2 = person2

    def __str__(self):
        return f"{self.person1}, is partners with {self.person2}"
    
    def __add__(self, other):
        person1 = self.person1 + other.person2
        person2 = other.person2 + self.person1
        return Group(person1, person2)


group1 = Group("Harry", "Ron")
group2 = Group("Hermoine", "Draco")
groups = group1 + group2

print(groups)