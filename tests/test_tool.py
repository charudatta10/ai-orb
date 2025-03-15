import pytest
from typing import Any, Dict, Optional
from ai_orb.tool import Tool  # Replace `your_module` with the actual module name

# Define sample functions to wrap with the Tool class
def sample_function(x: int, y: int) -> int:
    """
    Adds two numbers.
    """
    return x + y

def function_with_defaults(a: str = "default", b: int = 10) -> str:
    """
    Concatenates a string and a number.
    """
    return f"{a} {b}"

def function_with_no_args() -> str:
    """
    Returns a static string.
    """
    return "Hello, World!"

# Tests for the Tool class
def test_tool_initialization():
    """
    Test that the Tool initializes correctly with a function.
    """
    tool = Tool(func=sample_function)
    assert tool.name == "sample_function"
    assert tool.description == "Adds two numbers."

def test_tool_custom_name_and_description():
    """
    Test that the Tool initializes with a custom name and description.
    """
    tool = Tool(
        func=sample_function,
        name="CustomTool",
        description="This is a custom tool description."
    )
    assert tool.name == "CustomTool"
    assert tool.description == "This is a custom tool description."

def test_tool_call():
    """
    Test that the Tool correctly executes the wrapped function.
    """
    tool = Tool(func=sample_function)
    result = tool(3, 4)
    assert result == 7

def test_tool_call_with_defaults():
    """
    Test that the Tool correctly handles functions with default arguments.
    """
    tool = Tool(func=function_with_defaults)
    # Call with default arguments
    result = tool()
    assert result == "default 10"
    # Call with custom arguments
    result = tool(a="Custom", b=20)
    assert result == "Custom 20"

def test_tool_call_no_args():
    """
    Test that the Tool works with functions that take no arguments.
    """
    tool = Tool(func=function_with_no_args)
    result = tool()
    assert result == "Hello, World!"

def test_tool_error_handling():
    """
    Test that the Tool raises a ValueError for incorrect arguments.
    """
    tool = Tool(func=sample_function)
    with pytest.raises(ValueError, match="Error calling tool sample_function:"):
        tool(3)  # Missing required arguments

def test_tool_to_dict():
    """
    Test that the Tool correctly converts to a dictionary representation.
    """
    tool = Tool(func=sample_function)
    tool_dict = tool.to_dict()
    assert tool_dict["name"] == "sample_function"
    assert tool_dict["description"] == "Adds two numbers."
    assert callable(tool_dict["callable"])

# Run the tests
if __name__ == "__main__":
    pytest.main()
