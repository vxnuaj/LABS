''''
class: rec, cube, square

let square be the parent class. 

use super(). lol.
'''

class Square:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.area = None

    def s_area(self):
        self.area = self.height * self.width
        return self.area

class Rectangle(Square):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.area = None
    
    def r_area(self):
        super().s_area() # you don't need to run the param, self
        return self.area
    
class Cube(Square):
    def __init__(self, height, width, depth):
        super().__init__(height, width)
        self.depth = depth
        self.volume = None

    def __call__(self):
        self.volume = self.height * self.width * self.depth
        return self.volume
    

if __name__ == "__main__":
    rec = Rectangle(2, 3)

    print(rec.r_area())

    print(rec.area)

    cube = Cube(2, 3, 3)

    print(cube())