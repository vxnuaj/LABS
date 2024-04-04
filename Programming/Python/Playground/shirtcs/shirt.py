import os
import sys
from PIL import Image, ImageOps

def main():
    inf = get_input()
    create_output(inf)


#GET THE INPUT
def get_input():
    try:
        if len(sys.argv) < 3:
            sys.exit("Too few command-line arguments")
        elif len(sys.argv) > 3:
            sys.exit("Too many command-line argumentns")
        elif not sys.argv[2].endswith(".png") and not sys.argv[2].endswith(".jpg"):
            sys.exit("Invalid output")
        elif sys.argv[1][-3:] != sys.argv[2][-3:]:
            sys.exit("Input and output have different extensions")
        elif len(sys.argv) == 3 and sys.argv[1].endswith(".jpg") and sys.argv[2].endswith(".jpg"):
            input_file = sys.argv[1]
            inf = Image.open(input_file)
            return inf
    except FileNotFoundError:
        sys.exit("Input does not exist")
    

#CREATE THE OUTPUT
def create_output(inf):
    output_filename = sys.argv[2]
    output_file = inf.copy()
    output_file = ImageOps.fit(output_file, size = (1200,1200))
    shirt = Image.open("shirt.png")
    shirt = shirt.resize((1200, 1200))
    output_file.paste(shirt, (0,0), mask = shirt)
    output_file.save(output_filename)


if __name__ == "__main__":
    main()