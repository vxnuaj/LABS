import emoji

def main():
	alias = input("Input: ")
	conversion  = convert(alias)
	print(f"Output:{conversion}")

def convert(alias):
	conversion = emoji.emojize(alias, language='alias')
	return conversion	

main()


