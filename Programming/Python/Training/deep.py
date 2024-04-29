life = input("What meaning of life, the universe, and everything? ")
life = life.lower()

print(life)

def correct(l):
	match l:
		case "42" | "forty-two" | "forty two":
			print("Yes")
		case _:
			print("No it's 42!")

correct(life)
	





