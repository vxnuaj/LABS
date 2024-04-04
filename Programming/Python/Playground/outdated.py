months= [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]


def main():
	input_date = input("Date: ")
	convert(input_date)

def convert(input_date):
	while True:
		try:
			month, day, year = (input_date.split("/"))
			f_month, f_day = int(month), int(day)
			if f_month <= 12 and f_day <=  31:
				f_month, f_day = f"{int(f_month):02d}", f"{int(f_day):02d}"
				print(f"{year}-{f_month}-{f_day}")
				break
			else:
				input_date = input("Date: ")
		except:
			input_date = input_date.replace(",", "")
			month, day, year = input_date.split(" ")
			f_day = int(day)
			if f_day <= 31:
				f_day = f"{int(day):02d}"
				month = months.index(month)
				print(f"{year}-{month+1}-{f_day}")
				break
			else:
				input_date = input("Date: ")
		
	return

main()













