- Rather than letting a method take the instance as the first argument, you can make it take in the class by using `cls` and `@classmethod`

- @classmethods have the power to change the class variables amongst all instances of a class.
    - You can do so by running class.classmethod(args) or running instance.classmethod(args), whether you run a classmethod from an instance or class directly, either works for changing the class variable amongst all instances.