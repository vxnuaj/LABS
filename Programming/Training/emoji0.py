def main():
	emoji = input(f"Hey! How do you feel? :) or :(? ")
	emoji = convert(emoji)
	print(emoji)
	return



def convert(emoji):
	emoji = emoji.replace(":)", "😀")
	emoji = emoji.replace(":(", "☹️ ")
	return emoji

main()


