import numpy as np


class Language:
    _instance = None 
    
    def __new__(cls, *args, **kwargs):
        if cls._instance == None: 
            cls._instance = super().__new__(cls)
        return cls._instance

'''language = Language("Python", 1991)
print(f"Language: {language.lang}")
print(f"Year: {language.year}")'''

#lang1 = Language()
#lang2 = Language()

#print(lang1 == lang2) # showing that we're restricted to creating only a single instance of a class, per the __new__ singleton mechanism.

# each time we try to instantiate a new instance, we get the original one back.

# to make this more solid to a user we can run an error message if a user tries to creaste 2 of them

class Language2:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance == None:
            cls._instance = super().__new__(cls)
            cls.greeting = 'Hello'
            return cls._instance
        else:
            raise ValueError("Already created an instance of language2!")
        
lang1 = Language2()

print(lang1.greeting)
