## Enscaptulation

- Allows to put restrictions on accessing class variables and methods directly to avoid accidental modifiation ofdata.
-  Within enscaptualted variables, the variable itself can only be changed by an object's method, typically a setter.
  
#### public meethods and attributes

- Public methods are acc essibel from inside and outside the scope of a class. ANy instance of the class will have access to the method no matter the situation.

#### private methods and attributes

- any attribute / class variable that begins with an underscore is defined as non-public and purely meant for internal usage (but nothing is physically stopping you from using it). 
- any attribute / class variable that begins with 2 underscores, their name is changed autoamtically by pythoin making it more difficult to access. It isn't accessible by it's regular name with 2 '__', nor it's name without '__' nor it's name with one '_'  
- The name of the attribute is given a name defined as "*_ClassName__attributeName*, this is called ***name mangling***
- It can by bypassed by using the mangled name to define / redefine the variable
- here, this attribute would then be only used for internal usage, this time restricted by the language itself.
- though it can be accessed outside of the class, through the use of @property functions that are defined inside of the class as well as @Var.setter.
    - though at this point, the private variable is being passed onto the public class variable implicitly defined by the @property
