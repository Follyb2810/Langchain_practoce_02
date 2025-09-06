## 🔑 1. **Class Instance**

In Python, a class is a **blueprint**, and an **instance** is an object created from it.

```python
class Dog:
    def __init__(self, name):
        # constructor initializes instance attributes
        self.name = name
    
    def bark(self):
        print(f"{self.name} says Woof!")

# create instances (objects)
dog1 = Dog("Rex")
dog2 = Dog("Buddy")

dog1.bark()  # Rex says Woof!
dog2.bark()  # Buddy says Woof!
```

---

## 🔑 2. **Constructor (`__init__`)**

* `__init__` is the **constructor** (called automatically when you create an object).

```python
class Car:
    def __init__(self, brand, year):
        # instance variables (unique per object)
        self.brand = brand
        self.year = year

c = Car("Toyota", 2022)
print(c.brand)  # Toyota
```

---

## 🔑 3. **Public, Protected, Private**

Python does not enforce strict visibility like Java/C++. Instead, it uses **naming conventions**:

* **Public** (default): accessible anywhere
* **Protected (`_var`)**: convention that says "internal use only"
* **Private (`__var`)**: name mangling, harder to access directly

```python
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner          # public attribute
        self._balance = balance     # protected (convention only)
        self.__pin = "1234"         # private (name mangled)

    def deposit(self, amount):
        self._balance += amount     # can use protected internally

    def check_pin(self):
        return self.__pin

acct = BankAccount("Alice", 100)

print(acct.owner)       # ✅ Public → Alice
print(acct._balance)    # ⚠️ Works, but should be treated as protected
# print(acct.__pin)     # ❌ Error (private, name mangled)

print(acct.check_pin()) # ✅ Access via method → 1234
```

👉 Under the hood, `__pin` becomes `_BankAccount__pin`.

---

## 🔑 4. **Extends (Inheritance)**

Python uses **class inheritance** via parentheses:

```python
# Base (parent) class
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print("I make a sound")

# Derived (child) class extends Animal
class Dog(Animal):
    def __init__(self, name, breed):
        # call parent constructor
        super().__init__(name)
        self.breed = breed

    def speak(self):  # override parent method
        print(f"{self.name} says Woof!")

dog = Dog("Max", "Bulldog")
dog.speak()           # Max says Woof!
print(dog.breed)      # Bulldog
```

---

## 🔑 5. **Polymorphism (Bonus)**

Children can override parent methods → Python decides at runtime which method to call.

```python
animals = [Dog("Buddy", "Pug"), Animal("Generic")]

for a in animals:
    a.speak()   # Buddy says Woof!   |  I make a sound
```

---

✅ So to recap:

* **Instance** → object created from a class.
* **Constructor** → `__init__` method.
* **Public/Protected/Private** → naming conventions (`var`, `_var`, `__var`).
* **Extends** → inheritance using `class Child(Parent)`.
