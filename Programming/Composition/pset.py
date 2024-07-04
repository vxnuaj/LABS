class Character:
    def __init__(self, name, age, type):
        self.name = name
        self.age = age
        self.type = type
        self.armor = None
        self.weapon = None
        self.rings = None
        
    def add_armor(self, helm, chest, gauntlet, leggings):
        self.armor = (ArmorSet(helm, chest, gauntlet, leggings))

    def add_weapon(self, *weapons):
        self.weapon = (WeaponSet(*weapons))
   
  # where attr is a set of traits for a given ring (weight, name) 
    def add_rings(self, *attrs):
        rings = [Ring(*attr) for attr in attrs]
        self.rings = RingSet(rings)

class Weapon:
    def __init__(self, name, damage, durability, weight):
        self.name = name
        self.damage = damage
        self.durability = durability
        self.weight = weight

    def __str__(self):
        return f"Name: {self.name} | Damage: {self.damage} | Durability: {self.durability} | Weight: {self.weight}"    
       
        
class WeaponSet:
    def __init__(self, *weapons):
        for i, weapon in enumerate(weapons, start=1):
            setattr(self, f'weapon_{i}', weapon)

    def __str__(self):
        weapon_str = []
        for attr in dir(self):
            if attr.startswith('weapon'):
                weapon_str.append(getattr(self, attr))
        return '\n'.join(map(str, weapon_str))  
        
class Armor:
    def __init__(self, defense, weight, resistance):
        self.defense = defense
        self.weight = weight
        self.resistance = resistance

    def __str__(self):
        return f"Defense: {self.defense} | Weight: {self.weight} | Resistance: {self.resistance}"   
    
class Helm(Armor):
    def __init__(self, defense, weight, resistance):
        super().__init__(defense, weight, resistance)
    
        
class Chests(Armor):
    def __init__(self, defense, weight, resistance):
        super().__init__(defense, weight, resistance) 

class Leggings(Armor):
    def __init__(self, defense, weight, resistance):
        super().__init__(defense, weight, resistance)

class Gauntlets(Armor):
    def __init__(self, defense, weight, resistance):
        super().__init__(defense, weight ,resistance)

class ArmorSet:
    def __init__(self, helm, chest, gauntlet, leggings):
        self.helm = helm
        self.chest = chest
        self.gauntlet = gauntlet
        self.leggings = leggings 

    def __str__(self):
        return f"Helm: {self.helm}\nChest: {self.chest}\nGauntlets: {self.gauntlet}\nLeggings: {self.leggings}"
    
class Ring:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

class RingSet:
    def __init__(self, *rings):
        if len(rings) > 4:
            raise ValueError("can only be 4 rings max!")
        for i, ring in enumerate(rings, start = 1):
            setattr(self, f'ring{i}', ring)
        
if __name__ == "__main__":
    
    vxnuaj = Character('juan', 18, 'Knight')
    
    ''' creating armor ''' 
   
    helm = Helm(10, 20, 10)
    chests = Chests(5, 10, 5)
    leggings = Leggings(10, 20, 10)
    gauntlets = Gauntlets(2, 4, 2)
    
    vxnuaj.add_armor(helm, chests, gauntlets, leggings)
    
    print(vxnuaj.armor)
   
    '''creating weapons'''
   
    exile_greatsword = Weapon('Exile Greatsword', 20, 100, 50) 
    irithyll_ss = Weapon('Irithyll Straight Sword', 10, 50, 10)
    vxnuaj.add_weapon(exile_greatsword, irithyll_ss)
   
    print(vxnuaj.weapon)
    
    ''' creating rings '''
    
    