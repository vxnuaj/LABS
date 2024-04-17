import inflect
import sys
import datetime
from datetime import date

def main():
    year, month, day = input("Date of Birth: ").split('-')
    dob = dob_date(year, month, day)
    age_min = min_age(dob)
    date_words = convert_age_min(age_min)
    print(f"{date_words.capitalize()} minutes")
    return

def dob_date(year, month, day):
    try:
        year, month, day = int(year), int(month), int(day)
        dob = datetime.date(year, month, day)
        return dob
    except ValueError:
        sys.exit()

def min_age(dob):
    date_today = date.today()
    timesince = date_today - dob
    age_min = int(timesince.total_seconds() / 60)
    return age_min

def convert_age_min(age_min):
    p = inflect.engine()
    date_words = p.number_to_words(age_min, andword=',')
    return date_words


if __name__ == "__main__":
    main()


'''
- Apply error-correcting mechanism.
- Assume that the user was born at midnight and assume that it's midnight when the program is being run.
'''