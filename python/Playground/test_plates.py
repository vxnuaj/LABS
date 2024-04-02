import pytest
from plates import start_with, min_max, start_end, punc_space, is_valid

i = "CS50"

def test_starts_with_s():
    global i
    assert start_with(i) == True

def test_min_max():
    global i
    assert min_max(i) == True

def test_start_end():
    global i
    assert start_end(i) == True

def test_punc_space():
    global i
    assert punc_space(i) == True

def test_is_valid():
    global i
    assert is_valid(i) == True