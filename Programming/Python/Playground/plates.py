import string


def main():
    plate = input("Plate: ")
    if is_valid(plate):
        print("Valid")
    else:
        print("Invalid")


def is_valid(s):
	if start_with(s) and min_max(s) and start_end(s) and punc_space(s):
		return True
	else:
		return False



def start_with(s):
	if s[:1].isalpha():
		return True
	else:
		return False	

def min_max(s):
	if 2 <= len(s) <= 6:
		return True
	else:
		return False

def start_end(s):
	if s[:2].isalpha() and s[-2:].isdigit():
		return True
	else:
		return False
	
# Works Fine

def punc_space(s):
	if not any(char in string.punctuation for char in s):
		if not s.isspace():
			return True
		else:
			return False
		
if __name__ == "__main__":
	main()
