### Statc Methods

- Static methods can be called without an instance of a class
- Static methods don't requrie a self parameter
- Static methods don't require an instance of a class to be initialized
- Static methods cannot access nor modify class or instance attributes
- They behave like regular functions, just bound within a class, while not having anything to do with the class itself in terms of interaction.

Example:

```
class MyClass:
    @staticmethod
    def my_static_method(x, y):
        return x + y

result = MyClass.my_static_method(2, 3)  # Output: 5
```




