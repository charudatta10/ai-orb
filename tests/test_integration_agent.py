import pytest
from ai_orb.agent import Agent, Tool


# Mock LLM and Tools
class MockLLM:
    def generate(self, model: str, prompt: str):
        # Simulated LLM response
        return {"response": '[{"tool": "sample_tool_agent_1", "input": {"data": "test data"}}]'}

def sample_tool_agent_1(input_data):
    """Processes input and returns a result."""
    return f"Agent 1 processed {input_data}"

def sample_tool_agent_2(input_data):
    """Processes input and returns a result."""
    return f"Agent 2 processed {input_data}"


# Integration Test: Single Agent Execution
def test_single_agent_execution():
    tools = {"sample_tool_agent_1": sample_tool_agent_1}
    llm = MockLLM()
    agent = CollaborativeAgent(llm=llm, tools=tools, name="Agent 1", description="Test Agent")

    goal = "Test Goal"
    initial_context = {"sample_tool_agent_1": None}
    result = agent.solve_goal(goal=goal, initial_context=initial_context, max_iterations=2)

    assert "sample_tool_agent_1" in result
    assert result["sample_tool_agent_1"] == "Agent 1 processed {'data': 'test data'}"


# Integration Test: Multi-Agent Collaboration
def test_multi_agent_collaboration():
    tools_agent_1 = {"sample_tool_agent_1": sample_tool_agent_1}
    tools_agent_2 = {"sample_tool_agent_2": sample_tool_agent_2}
    llm = MockLLM()

    agent_1 = CollaborativeAgent(llm=llm, tools=tools_agent_2, name="Agent 1", description="Collaborative Agent 1")
    agent_2 = CollaborativeAgent(llm=llm, tools=tools_agent_1, name="Agent 2", description="Collaborative Agent 2")

    goal = "Collaborative Task"
    initial_context = {"sample_tool_agent_1": None, "sample_tool_agent_2": None}

    # Agent 1 starts the collaboration
    result_1 = agent_1.solve_goal(goal=goal, initial_context=initial_context, max_iterations=2)
    assert "sample_tool_agent_2" in result_1
    assert result_1["sample_tool_agent_2"] == "Agent 2 processed {'data': 'test data'}"

    # Agent 2 refines the task
    result_2 = agent_2.solve_goal(goal=goal, initial_context=result_1, max_iterations=2)
    assert "sample_tool_agent_1" in result_2
    assert result_2["sample_tool_agent_1"] == "Agent 1 processed {'data': 'test data'}"
