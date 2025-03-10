import pytest
from typing import Any, Callable, Dict
from ai_orb.tool import LLMTool  # Replace with the actual module name

# Helper functions for testing
def example_function(a: int, b: int) -> int:
    """Example function for testing."""
    return a + b

def example_function_no_docstring(a: int, b: int) -> int:
    return a * b

def example_function_no_args() -> str:
    """Example function with no arguments."""
    return "Hello, World!"

# Test Initialization
def test_initialization():
    tool = LLMTool(example_function, name="add", description="Adds two numbers")
    assert tool.name == "add"
    assert tool.description == "Adds two numbers"
    assert tool.func == example_function

def test_initialization_defaults():
    tool = LLMTool(example_function)
    assert tool.name == "example_function"
    assert tool.description == "Example function for testing."

def test_initialization_no_docstring():
    tool = LLMTool(example_function_no_docstring)
    assert tool.name == "example_function_no_docstring"
    assert tool.description == "No description provided"

# Test __call__ Method
def test_call_method():
    tool = LLMTool(example_function)
    result = tool(2, 3)
    assert result == 5

# Test forward Method
def test_forward_method():
    tool = LLMTool(example_function)
    result = tool.forward(2, 3)
    assert result == 5

def test_forward_method_with_kwargs():
    tool = LLMTool(example_function)
    result = tool.forward(a=2, b=3)
    assert result == 5

def test_forward_method_invalid_args():
    tool = LLMTool(example_function)
    with pytest.raises(ValueError):
        tool.forward(2)  # Missing required argument 'b'

# Test to_dict Method
def test_to_dict_method():
    tool = LLMTool(example_function, name="add", description="Adds two numbers")
    tool_dict = tool.to_dict()
    assert tool_dict == {
        "name": "add",
        "description": "Adds two numbers",
        "callable": tool.forward
    }

# Test Edge Cases
def test_function_no_args():
    tool = LLMTool(example_function_no_args)
    result = tool()
    assert result == "Hello, World!"

def test_function_no_args_with_args():
    tool = LLMTool(example_function_no_args)
    with pytest.raises(ValueError):
        tool(1)  # Function takes no arguments