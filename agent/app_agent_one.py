from typing import Optional
from langchain_core.tools import tool
import json

llm_output_string = '{"x": 5, "y": 2}'
llm_output_dict = json.loads(llm_output_string)


@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y


@tool
def exponentiate(x: float, y: float, z: Optional[float] = None) -> float:
    """
    Raise 'x' to the power of 'y'.
    Example: if x=5 and y=2, result = 25
    """
    return x**y


@tool
def subtract(x: float, y: float, z: Optional[float] = 0.3) -> float:
    """
    Subtract 'x' from 'y'.
    Example: if y=5 and x=2, result = 3
    """
    return y - x


print(add)
print(exponentiate.func(**llm_output_dict))
