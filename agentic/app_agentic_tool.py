from langchain_core.tools import tool


@tool
def add(x: float, y: float) -> float:
    """Add two numbers together and return the result."""
    return x + y


@tool
def subtract(x: float, y: float) -> float:
    """Subtract y from x and return the result."""
    return x - y


@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers and return the result."""
    return x * y


@tool
def divide(x: float, y: float) -> float:
    """Divide x by y and return the result. Raises error if y = 0."""
    if y == 0:
        raise ValueError("Division by zero is not allowed.")
    return x / y


@tool
def exponentiate(x: float, y: float) -> float:
    """Raise x to the power of y and return the result."""
    return x**y
