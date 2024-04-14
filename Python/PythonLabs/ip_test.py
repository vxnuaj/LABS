import pytest
from email import usr_input, validate

def main():
    test_usr_input()
    return

def test_usr_input():
    ip = usr_input()
    assert ip is int
    return

def test_validate():
    cor_validation = validate(23, 23, 23, 255)
    err_validation = validate(256, 256, 256, 256)
    assert cor_validation is True

if __name__ == "__main__":
    main()