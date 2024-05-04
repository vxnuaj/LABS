import pytest
from ip import usr_input, validate

def test_usr_input():
    one, two, three, four = usr_input()
    assert one.is_integer()
    assert two.is_integer()
    assert three.is_integer()
    assert four.is_integer()

def test_validate():
    assert validate(200, 300, 300, 1000) == "False"
