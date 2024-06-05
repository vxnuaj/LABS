### Class Methods

- With class methods, you don't need the creation of a class instance to call it.
- Class methods can access class attributes and modify them but can't modify instance attributes via `self`
- They use a `cls` parameter instead of `self`
- They can be used as an alternative constructor method, where calling it returns an instance of a class with modifications from the default __init__

Example:

```
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod
    def from_birth_year(cls, name, birth_year):
        age = 2023 - birth_year
        return cls(name, age)

john = Person("John", 30)  # Create an instance using the __init__ constructor
print(john.name, john.age)  # Output: John 30

jane = Person.from_birth_year("Jane", 1990)  # Create an instance using the class method constructor
print(jane.name, jane.age)  # Output: Jane 33
```

