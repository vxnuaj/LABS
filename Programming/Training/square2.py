class Square:
    def __init__(self, height):
        if not height:
            raise ValueError("MISSING HEIGHT")
        self.height = height

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if value > 10:
            raise ValueError("CANNOT BE GREATER THAN 10!")
        self._height = value
        self._area = value ** 2

    @property
    def area(self):
        return self._area

    def __str__(self):
        return f"AREA OF THE SQUARE IS {self.area}"

def main():
    num = get_input()
    sq = Square(num)
    print(sq)

def get_input():
    num = input("ENTER THE HEIGHT OF YOUR SQUARE: ")
    return int(num)

if __name__ == "__main__":
    main()
