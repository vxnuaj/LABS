file = input("File name: ")


def main(f):
	if " " in f:
		print("Unrecognizable")
	elif f.endswith("gif"):
		print("image/gif")
	elif f.endswith("jpg"):
		print("image/jpeg")
	elif f.endswith("png"):
		print("image/png")
	elif f.endswith("pdf"):
		print("application/pdf")
	elif f.endswith("txt"):
		print("text/plain")
	elif f.endswith("zip"):
		print("application/zip")

main(file)



