
def main():
	time = prompt_time()
	meal_time(time)

def prompt_time():
	time = input("What's the time? ")
	time = time.replace(":","")
	time = int(time)
	return time

def meal_time(time):
	if 700 <= time <= 800:
		print("breakfast time")
	elif 1200 <= time <= 1300:
		print("lunch time")
	elif 1800 <= time <= 1900:
		print("dinner time")
	else:
		print("")

main()
	
