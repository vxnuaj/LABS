import pytest
from jar import Jar


def test_init():
    jar = Jar()
    assert jar.size == 0
    ...


def test_str():
    jar = Jar()
    assert str(jar) == ""
    jar.deposit(1)
    assert str(jar) == "🍪"
    jar.deposit(11)
    assert str(jar) == "🍪🍪🍪🍪🍪🍪🍪🍪🍪🍪🍪🍪"


def test_deposit():
    jar = Jar()
    jar.deposit(2)
    assert jar.size == 2
    jar.deposit(2)
    assert jar.size == 4
    with pytest.raises(ValueError):
        jar.deposit(11)
    assert jar.size == 4
    ...


def test_withdraw():
    jar = Jar()
    jar.deposit(5)
    jar.withdraw(1)
    assert jar.size == 4
    jar.withdraw(2)
    assert jar.size == 2
    with pytest.raises(ValueError):
        jar.withdraw(10)
    assert jar.size == 2
    ...