### Problem Set on Object-Oriented Concepts

#### Public vs. Private vs. Protected Attributes

1. Create a class `Person` with attributes `name`, `_age`, and `__address`.
    - `name` should be public.
    - `_age` should be protected.
    - `__address` should be private.

2. Implement methods to get and set each attribute in the appropriate manner.

#### Public vs. Private vs. Protected Methods

3. Extend the `Person` class with methods `update_age`, `_change_address`, and `__display_info`.
    - `update_age` should be public.
    - `_change_address` should be protected.
    - `__display_info` should be private.

4. Test accessing and invoking each method from within the class and outside the class. Ensure proper visibility and access control.

#### Data Hiding

5. Define a class `BankAccount` with attributes `balance` and `__account_number`.
    - `balance` should be public.
    - `__account_number` should be private.

6. Implement methods to deposit (`deposit_funds`) and withdraw (`withdraw_funds`) money from the account, ensuring proper validation and data hiding.

#### Access Control (Using `__` for Private Attributes)

7. Create a class `Car` with attributes `make`, `__model`, and `_year`.
    - `make` should be public.
    - `__model` should be private.
    - `_year` should be protected.

8. Implement methods to get details (`get_make_model_year`) of the car, ensuring access control and proper encapsulation of private attributes.

### Additional Instructions

- For each problem, provide test cases that demonstrate your understanding of how each attribute or method behaves with respect to visibility and access control.
- Ensure that methods and attributes are accessed from both within the class and from external instances or functions to validate their behavior.
- Consider edge cases and exceptions handling where appropriate.

