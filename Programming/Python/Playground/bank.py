def main():
	g = input("Greeting: ")
	money_owed = value(g)
	print(f"${money_owed}")
	return


def value(g):
	g = g.lower().strip()
	if g.startswith("hello"):
		return 0
	elif g.startswith("h"):
		return 20
	else:
		return 100

if __name__ == "__main__":
	main()
	




