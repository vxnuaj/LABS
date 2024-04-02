def main():
	mass = input("Enter your mass in KG:")
	joules =convert(mass)
	print(joules)
	return

def convert(m):
	m = int(m)
	joules =(m) * pow(300000000,2)
	return joules

main()
