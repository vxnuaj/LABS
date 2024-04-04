def main():
	user_input = user_inputs()
	snake_case = convert(user_input)
	print(snake_case)	

def user_inputs():
	user_input = input("ENTER: ")
	return user_input

def convert(user_input):
	for letter in range(len(user_input)):
		val = user_input[letter].isupper()
		if val is True:
			beg = user_input[:letter]
			end = user_input[letter:].lower()
			snake_case = "_".join([beg,end])
			return snake_case
		else:
			snake_case = user_input
			return snake_case			
main()


# camelCase



