import random

class DragonBall:

    total_balls = 0

    def __init__(self, ball_number:int, is_collected:bool, location:str):
        self.ball_number = ball_number
        self.is_collected = is_collected
        self.location = location
        DragonBall.total_balls += 1

    def collect(self):
        num = random.randint(1, 10)
        if self.is_collected in [False, 'No']:
            if num >= 3:
                self.is_collected = True if self.is_collected is False else 'Yes'
                return f"Succesfully collected the {self.ball_number} star Dragon Ball!", True
            if num < 3:
                return f"Failed to collect the {self.ball_number} star Dragon Ball... enemies too powerful!", False
        elif self.is_collected in [True, 'Yes']:
            return f"Ball number {self.ball_number} is already collected!"

    def scatter(self, location:str):
        self._location = location
        return f"The {self.ball_number} star ★ Dragon Ball was scattered to {self._location}!"

    @classmethod
    def get_total_balls(cls):
        return f"There are a total of {cls.total_balls} balls in existence!", cls.total_balls
    
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
                self._is_collected = is_collected_str.title()
            else:
                raise ValueError("Invalid input, if str must be 'No' or 'Yes'! Else, should be bool!")
        elif isinstance(is_collected, bool):
            self._is_collected = is_collected
        else:
            raise ValueError("Invalid input! is_collected must be bool or str (Yes or No)!")
    
    @property
    def location(self):
        return self._location
    
    @location.setter
    def location(self, location):
        if isinstance(location, str):
            self._location = location.strip().title()
        else:
            raise ValueError("Location must be str!")
        
class DragonBallCollection:

    _total_balls = 0

    def __init__(self):
        self._balls = []

    def collect_ball(self, ball_number:int, is_collected: bool, location: str):
        ball = DragonBall(ball_number, is_collected, location)
        if ball.is_collected in [True, 'Yes']:
            self._balls.append(ball)
            DragonBallCollection._total_balls += 1
            return f"Successfully added the {ball.ball_number} star Dragon Ball to your collection!"
        elif ball.is_collected in [False, 'No']:
            ball_str, ball_bool = ball.collect()
            if ball_bool == True:
                self._balls.append(ball)
                DragonBallCollection._total_balls += 1
                return f"Succesfully collected the {ball.ball_number} star Dragon Ball and added to your collection!"
            elif ball_bool == False:
                return ball_str

    def remove_ball(self, ball_number):
        for ball in self._balls:
            if ball.ball_number == ball_number:
                self._balls.remove(ball)
                return f"Succesfully removed the {ball_number} star Dragon Ball from your collectionj!"
        return f"You never collected the {ball_number} ★ star Dragon Ball, can't remove anything!"
    
    def find_ball(self, ball_number):
        for ball in self._balls:
            if ball.ball_number == ball_number:
                return ball
        return f"You never collected the {ball_number} ★ star Dragon Ball, can't find it!"
    
    def scatter_ball(self, ball_number, location):
        for ball in self._balls:
            if ball.ball_number == ball_number:
                loc_str = ball.scatter(location)
                return loc_str
        return f"You never collected the {ball_number} ★ star Dragon Ball, can't scatter!"
    
    def find_by_location(self, location):
        ball_at_location = []
        location = location.title().strip()
        for ball in self._balls:
            if ball.location == location:
                ball_at_location.append(ball)
        if len(ball_at_location) == 0:
            return f"No balls at that location!"
        else:
            return ball_at_location
    
    @classmethod
    def get_total_balls(cls):
        return cls._total_balls

    @property
    def balls(self):
        return self._balls
    


db1 = DragonBall(10, 'No', 'San Francisco')

db1.collect()
db1.scatter('Asgard')

db1 = DragonBall(10, 'No', 'Olympus')

status = db1.get_total_balls()

print(status[0])

dbc = DragonBallCollection()

print(dbc.collect_ball(10, 'No', 'namek'))
print(dbc.collect_ball(5, True, 'namek'))
