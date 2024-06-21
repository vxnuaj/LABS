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
        'fullname': '    juan    vera ',
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
    
    data_list = [data, data2]
    
    characters = Character.load_characters(data_list)
    darksun = Game(characters)
    
    darksun.list_characters()