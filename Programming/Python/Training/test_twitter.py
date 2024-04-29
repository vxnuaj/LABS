import pytest
import sys
from twitter import vowel_strip, user_input

def test_strip():
        ui = "Minecraft Diamonds"
        strip_ui = vowel_strip(ui)
        vowels = ["i", "e", "a", "o", "u"]
        for vowel in vowels:
                assert vowel not in strip_ui.lower()

if __name__ == "__main__":
        test_strip()