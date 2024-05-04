import sys
from pyfiglet import Figlet
import random

def main():
	f = Figlet()
	cmd = sys.argv
	flist = f.getFonts()

	if len(cmd) == 2 or len(cmd) > 3:
		sys.exit("Invalid usage")
	elif len(cmd) == 3 and cmd[1] != "-f":
		sys.exit("Invalid usage")
	elif cmd[2] not in flist:
		sys.exit("Invalid usage")

	text = input("Input: ")
	sys_arg(f)
	print(f.renderText(text))
	return

def sys_arg(f):
	cmd = sys.argv
	figlet = Figlet()
	if len(cmd) == 3 and cmd[1] == "-f":
		font = cmd[2]
		f = f.setFont(font=font)
		return f
	elif len(cmd) == 1:
		font = random.choice(figlet.getFonts())
		f = f.setFont(font = font)
		return f

main()