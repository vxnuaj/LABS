### Problem: Dark Souls 3 Character Equipment Management

Create a `Character` class for managing a character's equipment in Dark Souls 3. Use composition to include classes for different types of equipment: `Weapon`, `ArmorSet`, and `RingSet`.

- **Weapon Class**: Manages a character's equipped weapon, including attributes like damage, type, and special effects.
  
- **ArmorSet Class**: Manages a set of armor pieces (head, chest, hands, legs) equipped by the character, with attributes for defense, weight, and resistance.

- **RingSet Class**: Manages a set of rings equipped by the character, each ring having unique effects on stats or abilities.

Your task is to design these classes using composition to allow a `Character` instance to manage and interact with their equipped weapons, armor sets, and rings effectively.

### Composition Problem

Design a system to simulate a digital clock using composition in Python. Implement the following classes:

- `Clock`: Represents the main digital clock.
  - Methods:
    - ~~`start()`: Starts the clock.~~
    - ~~`stop()`: Stops the clock.~~
    - ~~`set_time(hours, minutes, seconds)`: Sets the time on the clock.~~
    - ~~`tick()`: Advances the time by one second.~~
    - ~~`get_time()`: Returns the current time as a string (`HH:MM:SS`).~~

- `Display`: Represents the display part of the clock.
  - Attributes:
    - `hours`: Integer indicating hours (0-23).
    - `minutes`: Integer indicating minutes (0-59).
    - `seconds`: Integer indicating seconds (0-59).
  - Methods:
    - `show()`: Returns the current time in `HH:MM:SS` format.

Use composition to link these classes together, ensuring that the `Clock` class manages the overall functionality using an instance of the `Display` class to handle time representation.

Create instances of the `Clock` class, set the time, and demonstrate the clock ticking over a few seconds.

