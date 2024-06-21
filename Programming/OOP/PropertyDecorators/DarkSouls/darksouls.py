from typing import Union, List

class Character:

    def __init__(self, fullname:str, level:int, vigor:int, strength:int, dexterity:int, equipment:dict):
        self.firstname = None
        self.lastname = None
        self.fullname = fullname
        self.level = level
        self.vigor = vigor
        self.strength = strength
        self.dexterity = dexterity
        self.equipment = equipment
        
    def remove_equipment(self, name, amount):
        if not  isinstance(name, str):
            raise ValueError("Name must be type str!")
        if not isinstance(amount, int):
            raise ValueError("Amount must be type int!")
        
        name = ' '.join(name.split()).title()
        if self.equipment[name] < amount:
            self.equipment[name] = 0
        else:
            self.equipment[name] -= amount
            
        if self.equipment[name] == 0:
            self.equipment.pop(name)

    @classmethod
    def from_dict(cls, data:dict):
        try:
            fullname, level, vigor, strength, dexterity, equipment = data.values()
        except ValueError as e:
            if 'not enough values to unpack' in str(e):
                raise ValueError("Not enough items in data! Include fullname, level, vigor, strength, dexterity, and equipment in the respective order!")
            elif 'too many values to unpack' in str(e):   
                raise ValueError("Too many items in data! Only include fullname, level, vigor, strength, dexterity, and equipment in the respective order!")
        return cls(fullname, level, vigor, strength, dexterity, equipment)
    
    @classmethod
    def load_characters(cls, data_list: list[dict]) -> list:
        if not isinstance(data_list, list):
            raise ValueError('data_list must be type list!')
        characters = [cls.from_dict(i) for i in data_list]
        return characters
    
    @staticmethod
    def calculate_total_attributes(character):
        attr_points = character.vigor + character.strength + character.dexterity
        return attr_points
    
    @staticmethod
    def validate_character_data(data):
        attr = ['fullname', 'level', 'vigor', 'strength', 'dexterity', 'equipment']
                
        if not isinstance(data, dict):
            raise ValueError("data must be type dict!")
        
        for i in data: # Checking that all values belonging to a given key are the right type
            if not isinstance(i, str):
                raise ValueError("all keys in data dict must be type str!")
            elif i.lower() == 'fullname' and not isinstance(data[i], str):
                print(type(i))
                raise ValueError(f"{''.join(i.split())} must be type str!")
            elif i.lower() in ['level', 'vigor', 'strength', 'dexterity'] and not isinstance(data[i], int):
                raise ValueError(f"{i} must be type int!")
            elif i.lower() == 'equipment' and not isinstance(data[i], dict):
                raise ValueError(f"{i} must be type dict!")
                
        for i in data: # Checking that the ranges for levels and attributes are in the right range.
            if i.lower() == 'level' and not 1 <= data[i] <= 802:
                raise ValueError(f"Attribute {i} must be between range 1 to 802, inclusive!")
            elif i.lower() in ['vigor', 'strength', 'dexterity'] and not 1 <= data[i] <= 99:
                raise ValueError(f"Attribute {i} must be between range 1 to 99, inclusive!")
                
        char_attr = [attr.lower()for attr in data.keys()]

        for i in attr:
            if i not in char_attr:
                raise ValueError(f"Attribute {i} does not exist in the data dict!")
        if len(data) != len(attr):
            raise ValueError(f"Data must be {', '.join(attr)}. You have {', '.join(data.keys())}")
                
    '''

    @staticmethod
        def validate_character_data(data):
            attr = ['fullname', 'level', 'vigor', 'strength', 'dexterity', 'equipment']

            if not isinstance(data, dict):
                raise ValueError("data must be type dict!")
            elif len(data) != 6:
                raise ValueError("data must be 'fullname', 'level', 'vigor', 'strength', 'dexterity', and 'equipment'.")
                
            if not all(isinstance(i, str) for i in data):
                raise ValueError("all keys in data dict must be type str!")
            
                
            char_attr = [''.join(attr.split()).lower()for attr in data.keys()]

            for i in attr:
                if i not in char_attr:
                    raise ValueError(f"Attribute {i} does not exist in the data dict!")
                
            return True


    '''        


    @staticmethod
    def caluclate_level_up_cost(level:int):

        if not 1 <= level <= 802:
            raise ValueError("Level must be in range 1, 802, inclusive!")
        elif not isinstance(level, int):
            raise ValueError("Your soul level must be type int!")
        elif level == 802:
            return f"You're already at level {level}, the maximum possible level. Congratulations!"
        
        
        levels_12 = [673, 689, 706, 723, 740, 757, 775, 793, 811, 829, 847]
        
        if level < 13:
            level -= 1
            cost = levels_12[level]
        elif level >= 13:
            cost = int(((.002 * level) ** 3) + ((3.06 * level) ** 2) + (105.6 * level) - 895)
        return cost

    @property
    def fullname(self):
        return self._fullname
    
    @fullname.setter
    def fullname(self, fullname):
        if not isinstance(fullname, str):
            raise ValueError("All names must either be NoneType or type str!")
        fullname = ' '.join(fullname.split()).title()
        names = fullname.split(' ')
        if len(names) == 2:
            self._fullname = fullname
            self._firstname = names[0]
            self._lastname = names[1]
        elif len(names) == 1:
            self._fullname = fullname
            self._firstname = fullname
            self._lastname = None
        else:
            raise IndexError("Fullname is too long! Must only be first and last!")

    @fullname.deleter
    def fullname(self):
        self._fullname = None
        self._firstname = None
        self._lastname = None
        
    @property
    def firstname(self):
        return self._firstname
    
    @firstname.setter
    def firstname(self, first):
        if not isinstance(first, (str, type(None))):
            raise ValueError("All names must either be NoneType or type str!")
        elif isinstance(first, type(None)) == True:
            self._first = None
            return
        
        first = first.strip().title()
        first_name = first.split(' ')

        if len(first_name) > 1:
            raise ValueError("First name can only be 1 name!")
        self._firstname = first

        if self._lastname:
            self._fullname = f"{self._firstname} {self._lastname}"
        else:
            self._fullname = first

    @firstname.deleter
    def firstname(self):
        if self._lastname:
            self._fullname = self._lastname
            self._firstname = None
        else:
            self._fullname = None
            self._firstname = None

    @property
    def lastname(self):
        return self._lastname
    
    @lastname.setter
    def lastname(self, last):
        if not isinstance(last, (str, type(None))):
            raise ValueError("All names must either be NoneType or type str!")
        elif isinstance(last, type(None)) == True:
            self._lastname = None
            return

        last = last.strip().title()
        last_name = last.split(' ')
        
        if len(last_name) > 1:
            raise ValueError("Last name can only be 1 name!")
        self._lastname = last
        
        if self._firstname:
            self._fullname = f"{self._firstname} {self._lastname}"
        else:
            self._fullname = last

    @lastname.deleter
    def lastname(self):
        if self._firstname:
            self._fullname = self._firstname
            self._lastname = None
        else:
            self._fullname = None
            self._lastname = None

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        if not isinstance(level, int):
            raise ValueError("Character level must be type int!")
        elif level not in range(803):
            raise ValueError("Level must be between 0 and 802, inclusive!")
        self._level = level
        
    @level.deleter
    def level(self):
        self._level = 0

    @property
    def vigor(self):
        return self._vigor
    
    @vigor.setter
    def vigor(self, vigor):
        if not isinstance(vigor, int):
            raise ValueError("Vigor must be type int!")
        elif vigor not in range(100):
            raise ValueError("Vigor must be between 0 and 99, inclusive!")
        self._vigor = vigor

    @vigor.deleter
    def vigor(self):
        self._vigor = 0

    @property
    def strength(self):
        return self._strength
    
    @strength.setter
    def strength(self, strength):
        if not isinstance(strength, int):
            raise ValueError("Strength must be type int!")
        elif strength not in range(100):
            raise ValueError("Strength must be betwen 0 and 99, inclusive!")
        self._strength = strength

    @strength.deleter
    def strength(self):
        self._strength = 0

    @property
    def dexterity(self):
        return self._dexterity

    @dexterity.setter
    def dexterity(self, dexterity):
        if not isinstance(dexterity, int):
            raise ValueError("Dexterity must be type int!")
        elif dexterity not in range(100):
            raise ValueError("Strength must be betwen 0 and 99, inclusive!")
        self._dexterity = dexterity

    @dexterity.deleter
    def dexterity(self):
        self._dexterity = 0

    @property
    def equipment(self):
        return self._equipment
    
    @equipment.setter
    def equipment(self, equipment):
        if not isinstance(equipment, dict):
            raise ValueError("Equipment must be type dict!")
        for item, amt in equipment.items():
            if not isinstance(item, str):
                raise ValueError(f"Item name must be type str, denoting the name of a given item that {self.fullname} holds!")
            if not isinstance(amt, int):
                raise ValueError(f"Value associated with {item} must be type int, denoting the amount {self.fullname} holds!")
            if not amt > 0:
                raise ValueError(f"Value associated with {item} must be greater than 0! You can't carry nothing if you have something!")
            self._equipment = {item.strip().replace("'", "œ").title().replace("œ", "'"): amt for item, amt in equipment.items()}

    @equipment.deleter
    def equipment(self):
        self._equipment.clear()
        
class Game:
    def __init__(self, characters: Union[object, list[object]]):
        self.characters = []
        self.__add_character(characters)

    def add_character(self, character:Union[object, list[object]]):
        if isinstance(character, object):
            self.characters.append(character)
        elif isinstance(character, list):
            if not all(isinstance(i, object) for i in character):
                    raise ValueError("Not all items in your character list are characters!")
            self.characters += [i for i in character]

    def __add_character(self, character:Union[object, list[object]]):
        if isinstance(character, Character):
            self.characters.append(character)
        elif isinstance(character, list):
            if not all(isinstance(i, object) for i in character):
                    raise ValueError("Not all items in your character list are characters!")
            self.characters.extend(character)
        return self.characters

    def remove_character(self, fullname):
        if not isinstance(fullname, str):
            raise ValueError("fullname must be type str!")
    
        fullname = ' '.join(fullname.split()).title()
        self.characters = [i for i in self.characters if i.fullname != fullname]
        
    def list_characters(self):
        i = 1
        num = len(self.characters)
        print(f"{num} TOTAL CHARACTERS:")
        for character in self.characters:
            print(f"{i}. {character.fullname}")
            i += 1
        
    

if __name__ == "__main__":

    fullname = '    juan    vera '
    level = 802
    vigor = 99
    strength = 99
    dexterity = 99
    equipment = {
        'exile greatsword': 1,
        "dancer's enchanted swords" : 1,
        'pyromancy flame': 1
    }


    data = {
        'fullname': 'vxnuaj',
        'level': 802,
        'vigor': 99,
        'strength': 99,
        'dexterity': 99,
        'equipment': {
            'exile greatsword': 1,
            "dancer's enchanted swords" : 1,
            'pyromancy flame': 1
    }
    }
    
    data2 = {
        'fullname': '    tfue',
        'level': 801,
        'vigor': 94,
        'strength': 91, 
        'dexterity': 89,
        'equipment': {
            'peensword': 1,
            'bush': 3,
            'prymancy flame': 1
        }
        
    }
    

    
vxnuaj = Character.from_dict(data)

print(Character.calculate_total_attributes(vxnuaj))