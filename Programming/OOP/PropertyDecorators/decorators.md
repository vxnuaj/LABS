- Property decorators allow us to define a method using @property tag and access it like a class attribute

- @property allows us to call the method as an attribute but doesn't allow us to set the attribute itself. to do so, we'd need to introduce an @setter.

- @property takes in 2 parameters / arguments

```

@property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    @fullname.setter
    def fullname(self, fullname):
        first, last = fullname.split(' ')
        self.first = first
        self.last = last

```

in `@fullname.setter`, the second parameter represents the parameter that's used when assigning to the attritbute

`person.fullname = 'Jane Smith`, the second parameter being passed to the method defiend under `@fullname.setter` is 'Jane Smith', which is then used to modify any other attributes that depend on the fullname method / attribute (given per `@property`)

finally, `@deleter` allows for us to set the rules for what happen when we call `del` to delete a variable, for example as:

```
del fullname
```

we can define a deleter, for example, as:

```
@fullname.deleter
    def fullname(self):
        self.first = None
        self.last = None
        print("Deleted employee name!")

```