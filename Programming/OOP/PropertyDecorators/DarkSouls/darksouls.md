### Problem Statement: Dark Souls 3 Enhanced Character Management System

You are tasked with creating an enhanced Character Management System for the game Dark Souls 3 in Python. The system should manage characters, their attributes, equipment, and the game's level-up system. Implement the following classes and methods:

#### Class: `Character`

1. **Attributes:**
   - ~~`name` (string): The name of the character.~~
   - ~~`level` (integer): The level of the character.~~
   -~~`vigor` (integer): The vigor stat of the character.~~
   - ~~`strength` (integer): The strength stat of the character.~~
   - ~~`dexterity` (integer): The dexterity stat of the character.~~
   - ~~`equipment` (dictionary): A dictionary containing equipment slots (e.g., 'weapon', 'armor') and their corresponding items.~~

2. **Methods:**
   - ~~`__init__(self, name, level, vigor, strength, dexterity, equipment)`: Initializes the character with the provided attributes and equipment.~~
   - ~~[X] CREATE A METHOD THAT GETS RID OF ITEMS IN EQUIPMENT BASED ON THE NAME AND AMOUNT PARAMTERS~~
   
3. **Properties:**
   - ~~`name`: Property for the character's name with getter, setter, and deleter.~~
   - ~~`level`: Property for the character's level with getter, setter, and deleter~~ 
   - ~~`vigor`: Property for the character's vigor with getter, setter, and deleter.~~
   - ~~`strength`: Property for the character's strength with getter, setter, and deleter.~~
   - ~~`dexterity`: Property for the character's dexterity with getter, setter, and deleter.~~
   - ~~`equipment`: Property for the character's equipment with getter, setter, and deleter.~~

#### Class: `Game`

1. **Attributes:**
   - ~~`characters` (list): A list of `Character` instances representing the characters in the game.~~

2. **Methods:**
   - ~~`__init__(self)`: Initializes the game with an empty list of characters.~~
   - `~~add_character(self, character)`: Adds a `Character` instance to the `characters` list.~~
   - ~~`remove_character(self, name)`: Removes a `Character` instance from the `characters` list by name.~~
   - ~~`list_characters(self)`: Lists all characters in the game.~~

3. **Class Methods:**
   - ~~`from_dict(cls, data)`: Class method that creates a `Character` instance from a dictionary containing character attributes.~~
   - ~~`load_characters(cls, data_list)`: Class method that creates a list of `Character` instances from a list of dictionaries containing character attributes.~~

      > In essence, create a list of characters (in object form), from a list of dictoinaries where each dictionary holds the attributes of each character.

4. **Static Methods:**
   - `calculate_level_up_cost(level)`: Static method that calculates the soul cost to level up based on the current level.
   - `validate_character_data(data)`: Static method that validates if the provided character data (dictionary) contains all necessary attributes.
   - `calculate_total_attributes(character)`: Static method that calculates the total attribute points of a character.

Implement the classes and methods described above. Make sure to include necessary error handling and input validation where appropriate. This will help you practice using staticmethods, classmethods, properties with getters, setters, and deleters, and other OOP concepts in Python.
