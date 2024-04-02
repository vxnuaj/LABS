def main():
	x, y, z = info()
	result = compute(x,y,z)
	print(result)	

def info():
	calc_input = input("ENTER YOUR OPERARION: ")
	calc_input = calc_input.split(" ")
	x, y, z = float(calc_input[0]), calc_input[1], float(calc_input[2])
	#x = float(input("ENTER THE FIRST NUMBER: "))
	#y = (input("ENTER THE OPERATION: "))
	#z = float(input("ENTER THE SECOND NUMBER: "))
	return x, y, z

def compute(x,y,z):
	if y == "+":
		return add(x,z)
	elif y == "-":
		return subtract(x,z)
	elif y == "*":
		return multiply(x,z)
	elif y == "/":
		return divide(x,z)	

def add(x, z):
	return x + z

def subtract(x, z):
	return x - z

def multiply(x, z):
	return x * z


def divide(x, z):
	return x  / z

main()



