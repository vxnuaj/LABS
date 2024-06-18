### DragonBall Class

Create a class `DragonBall` that represents a single Dragon Ball. The class should have the following requirements:

- The class should have properties for `ball_number`, `is_collected`, and `location`.
- The `ball_number` property should represent the number of the Dragon Ball (e.g., 1 for the 1-star ball, 2 for the 2-star ball, and so on).
- The `is_collected` property should be a boolean indicating whether the Dragon Ball has been collected or not.
- The `location` property should represent the current location of the Dragon Ball.
- The class should have a method `collect()` that sets the `is_collected` property to `True`.
- The class should have a method `scatter(location)` that sets the `location` property to a new location.
- The class should have a class variable `total_balls` that stores the total number of Dragon Balls in existence.
- The class should have a class method `get_total_balls()` that returns the total number of Dragon Balls in existence.

### DragonBallCollection Class

Create a class `DragonBallCollection` that manages a collection of Dragon Balls. The class should have the following requirements:

- The class should have a property `balls` that stores a list of `DragonBall` instances.
- The `balls` property should be a read-only property using the `@property` decorator.
- The class should have a method `add_ball(ball_number, is_collected, location)` that creates a new `DragonBall` instance with the given parameters and adds it to the collection.
- The class should have a method `remove_ball(ball_number)` that removes the `DragonBall` instance with the given ball number from the collection.
- The class should have a method `find_ball(ball_number)` that returns the `DragonBall` instance with the given ball number.
- The class should have a method `scatter_ball(ball_number, location)` that scatters the `DragonBall` instance with the given ball number to the specified location.
- The class should have a class method `get_total_balls()` that returns the total number of Dragon Balls in the collection.
- The class should have a class method `find_by_location(location)` that returns a list of `DragonBall` instances that are currently located at the specified location.

