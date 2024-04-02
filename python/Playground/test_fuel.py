import pytest
from fuel import convert, gauge

fraction = "3/7"
percentage = 42.86

def test_convert():
    global fraction
    assert isinstance(convert(fraction), float)
    assert 0 < convert(fraction) < 100

def test_gauge():
    global percentage
    if 1 < percentage < 99:
        assert gauge(percentage) == f"{percentage}%"
    elif percentage <= 1:
        assert gauge(percentage) == "E"
    elif percentage >= 99:
        assert gauge(percentage) == "F"