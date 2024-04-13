import sys 

def main():
	fraction = input("Fraction: ")
	percentage = convert(fraction)
	status = gauge(percentage)
	print(status)

def convert(fraction):
	try:
		x, y = float(fraction[0]), float(fraction[2])
		percentage = ((x/y) * 100)
		return percentage
	except ValueError:
		sys.exit("Invalid input!")

def gauge(percentage):
	if percentage >= 99:
		return "F"
	elif percentage <= 1:
		return "E"
	else:
		return f"{percentage}%"


if __name__ == "__main__":	
	main()




