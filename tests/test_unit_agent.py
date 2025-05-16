import pytest
from typing import Dict, Any
from src.sandbox import SecureSandbox
from src.agent import Think, Act, Observe, Agent, Tool


# Mock for ollama client
class MockLLM:
    def generate(self, model: str, prompt: str):
        # Simulated LLM response
        return {"response": '[{"tool": "sample_tool_agent_1", "input": {"data": "test data"}}]'}


# Mock Tools
def mock_tool_1(input_data: Dict[str, Any]) -> str:
    """Mock tool 1: Processes input and returns a result."""
    return f"Processed by Tool 1: {input_data}"

def mock_tool_2(input_data: Dict[str, Any]) -> str:
    """Mock tool 2: Processes input and returns a result."""
    return f"Processed by Tool 2: {input_data}"


# Unit Test: Think Module
def test_think_generate_plan():
    tools = {"mock_tool_1": Tool(func=mock_tool_1), "mock_tool_2": Tool(func=mock_tool_2)}
    llm = MockLLM()
    think = Think(llm=llm, tools=tools)
    goal = "Test Goal"
    context = {"current_state": "test"}

    plan = think.generate_plan(goal=goal, context=context)
    assert isinstance(plan, list)
    assert len(plan) > 0
    assert plan[0]["tool"] == "sample_tool_agent_1"
    assert plan[0]["input"] == {"data": "test data"}


# Unit Test: Act Module
def test_act_execute_tool():
    tools = {"mock_tool_1": mock_tool_1}
    sandbox = SecureSandbox(tools=tools)
    act = Act(sandbox=sandbox)

    # Test valid tool execution
    result = act.execute_tool("mock_tool_1", {"data": "input value"})
    assert result["success"] is True
    assert result["output"] == "Processed by Tool 1: {'data': 'input value'}"

    # Test invalid tool execution
    result = act.execute_tool("non_existent_tool", {"data": "input value"})
    assert result["success"] is False
    assert "error" in result


# Unit Test: Observe Module
def test_observe():
    observe = Observe()
    observation = {"step": "test_step", "result": "test_result"}
    plan = [{"tool": "test_tool", "input": {"data": "test"}}]

    # Test recording observation
    observe.record_observation(observation)
    assert len(observe.observations) == 1
    assert observe.observations[0] == observation

    # Test recording plan
    observe.record_plan(plan)
    assert len(observe.past_plans) == 1
    assert observe.past_plans[0]["plan"] == plan


# Unit Test: Tool Class (Optional, if applicable)
def test_tool_class():
    tool = Tool(func=mock_tool_1, name="MockTool1", description="Test Description")
    assert tool.name == "MockTool1"
    assert tool.description == "Test Description"
    assert tool({"data": "test input"}) == "Processed by Tool 1: {'data': 'test input'}"
