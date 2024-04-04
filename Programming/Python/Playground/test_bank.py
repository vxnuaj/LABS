import pytest
from bank import value

def main():
    test_Hello()
    test_h()
    test_otherwise()

def test_Hello():
    g = "Hello"
    assert value(g) == 0

def test_h():
    g = "heya kid"
    assert value(g) == 20

def test_otherwise():
    g = "yo what's up man!"
    assert value(g) == 100

if __name__ == "__main__":
    main()