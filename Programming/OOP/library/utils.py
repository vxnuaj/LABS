import sys

class Utils:
    def __init__(self):
        pass
    
    @staticmethod
    def list_to_string(items):
        str = ''
        if isinstance(items, list) and items:
            try:
                for item in items:
                    str += item
                if str == '':
                    raise ValueError("List is empty!")
            except TypeError: 
                raise ValueError("All items in list must be a string!")
            return str
        elif not items:
            return None
        else:
            raise TypeError("Input must be a list!")
        
    @staticmethod
    def dict_to_string(items):
        pass