import pytest
import re
from seasons import dob_date, min_age, convert_age_min


def test_dob_date():
    dob = dob_date("2023", "09", "30")
    assert re.search(r"^\d{4}\-(0[1-9]|1[012])\-(0[1-9]|[12][0-9]|3[01])$", str(dob))

def test_min_age():
    dob = dob_date("2023", "09", "30")
    assert isinstance(min_age(dob), int)

def test_convert_age_min():
    dob = dob_date("2023", "09", "30")
    age_min = min_age(dob)
    assert isinstance(convert_age_min(age_min), str)