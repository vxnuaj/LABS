import time

class MyMetaclass(type):
    
    base_time = time.perf_counter()
    
    def __new__(mcs, name, bases, namespace):
        print(mcs, name, bases, namespace)
        namespace['__class_load_time__'] = time.perf_counter() - MyMetaclass.base_time 
        return super().__new__(mcs, name, bases,namespace)
    
    '''
   
    This is creating a meta class. 
    Whenever we create a metaclass, we have to inherit from `type`.
    This is as `type` has the blueprint for creating a class.
    Even the type class is created from within itself, type
    
    '''
    pass

class A(metaclass=MyMetaclass):
    
    '''
    This is creating a child of the meta class.
    At defualt, a class uses the `type` class and it's __new__ method to create it.

    But defining the metaclass as `MyMetaClass` uses the `MyMetaClass` __new__ 
    method, whether it's the defualt __new__ or a modified version of it.

    ''' 
    
    pass

def main():
    a =  A() 
    print(f"{type(a)=}") # will be initiated by the A() class, through it's __new__
    print(f"{type(A)=}") # will be initiated by the MyMetaClass, through it's __new__
    print(f"{A.__class_load_time__} after base time")
    
hello = main() 