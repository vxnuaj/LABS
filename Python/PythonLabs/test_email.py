import pytest
from ip import usr_input, validate

def main():
    test_usr_input()
    test_validate()
    return

def test_usr_input():
    one, two, three, four = usr_input()
    assert one.is_integer()
    assert two.is_integer()
    assert three.is_integer()
    assert four.is_integer()

def test_validate():
    cor_validation = validate(23, 23, 23, 255)
    err_validation = validate(256, 256, 256, 256)
    assert err_validation == "True"

if __name__ == "__main__":
    main()