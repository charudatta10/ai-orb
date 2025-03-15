import pytest
from ai_orb.tool import Tool  # Replace with the actual module name
from ai_orb.sandbox import SecureSandbox  # Replace with the actual module name

# Helper functions for testing
def safe_function(a: int, b: int) -> int:
    """A safe function for testing."""
    return a + b

def dangerous_function():
    """A dangerous function for testing."""
    import os  # This should be blocked
    return os.getcwd()

# Test Initialization
def test_initialization_defaults():
    sandbox = SecureSandbox()
    assert sandbox.max_time == 5
    assert sandbox.max_output_size == 10 * 1024
    assert sandbox.allowed_modules == set()

def test_initialization_custom():
    sandbox = SecureSandbox(max_time=10, max_output_size=2048, allowed_modules=["math"])
    assert sandbox.max_time == 10
    assert sandbox.max_output_size == 2048
    assert sandbox.allowed_modules == {"math"}

# Test Code Validation
def test_validate_code_safe():
    sandbox = SecureSandbox()
    code = "x = 1 + 2"
    sandbox.validate_code(code)  # Should not raise an error

def test_validate_code_dangerous_import():
    sandbox = SecureSandbox()
    code = "import os"
    with pytest.raises(ValueError, match="Import of 'os' is not allowed"):
        sandbox.validate_code(code)

def test_validate_code_dangerous_function():
    sandbox = SecureSandbox()
    code = "eval('1 + 1')"
    with pytest.raises(ValueError, match="Call to 'eval' is not allowed"):
        sandbox.validate_code(code)

# Test Code Execution
def test_execute_code_safe():
    sandbox = SecureSandbox()
    code = "x = 1 + 2"
    result = sandbox.execute(code)
    assert result["success"] is True
    assert result["output"] == ""
    assert result["error"] == ""

def test_execute_code_dangerous():
    sandbox = SecureSandbox()
    code = "import os"
    result = sandbox.execute(code)
    assert result["success"] is False
    assert "Import of 'os' is not allowed" in result["error"]

def test_execute_code_timeout():
    sandbox = SecureSandbox(max_time=1)
    code = "while True: pass"
    result = sandbox.execute(code)
    assert result["success"] is False
    assert "exceeded 1 second time limit" in result["error"]

# Test Function Execution
def test_execute_function_safe():
    sandbox = SecureSandbox()
    wrapped_function = sandbox.execute(safe_function)
    result = wrapped_function(2, 3)
    assert result == 5

def test_execute_function_dangerous():
    sandbox = SecureSandbox()
    with pytest.raises(ValueError, match="Function validation failed"):
        sandbox.execute(dangerous_function)

# Test Integration with LLMTool
def test_llmtool_integration():
    sandbox = SecureSandbox()
    tool = LLMTool(safe_function, name="safe_tool", description="A safe tool")
    wrapped_tool = sandbox.execute(tool)
    result = wrapped_tool(2, 3)
    assert result == 5

def test_llmtool_integration_dangerous():
    sandbox = SecureSandbox()
    tool = LLMTool(dangerous_function, name="dangerous_tool", description="A dangerous tool")
    with pytest.raises(ValueError, match="Function validation failed"):
        sandbox.execute(tool)