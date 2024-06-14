class DragonBall:
    def __init__(self, ball_number:int, is_collected:bool, location:str):
        self.ball_number = ball_number
        self.is_collected = is_collected
        self.location = location
    
    @property
    def ball_number(self):
        return self._ball_number
    
    @ball_number.setter
    def ball_number(self, ball_number):
        if isinstance(ball_number, int) == False:
            raise ValueError("ball_number must be an integer!")
        else:
            self._ball_number = ball_number

    @property
    def is_collected(self):
        return self._is_collected
    
    @is_collected.setter
    def is_collected(self, is_collected):
        if isinstance(is_collected, str):
            is_collected_str = is_collected.strip()
            if is_collected_str.lower() in ['no', 'yes']:
                self._is_collected = is_collected_str
            else:
                raise ValueError("Invalid input, if str must be 'No' or 'Yes'! Else, should be bool!")
        elif isinstance(is_collected, bool):
            self._is_collected = is_collected
        else:
            raise ValueError("Invalid input! is_collected must be bool or str (Yes or No)!")
    
    @property
    def location(self):
        return self.location
    
    @location.setter
    def location(self, location):
        if isinstance(location, str):
            self._location = location.strip()
        else:
            raise ValueError("Location must be str!")
        
db1 = DragonBall(10, 'Yes', 'Asgard')