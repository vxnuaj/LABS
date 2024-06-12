first_name = input("What is your first name?")
last_name = input("What is your last name?")


first_name = first_name.title().strip()
last_name = last_name.title().strip()

print('{} {}'.format(first_name, last_name))