import numpy as np

class Test:
    def __init__(self, val):
        self.val = val
      
    def method(self):
        print(self.val)
        print(self.val2)
       
    @property
    def val(self):
        return self._val
  
    @val.setter
    def val(self, val):
        self._val = val * 2 
        self.val2 = np.random.randn()
       
       
if __name__ == "__main__":
    test = Test(val =1)
    test.method() 
    