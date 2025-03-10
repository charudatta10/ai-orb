import pytest
from unittest.mock import MagicMock

from ai_orb.agent import LLMReasoner, AgentMemory, GoalOrientedAgent

def test_generate_plan():
    """Test LLMReasoner.generate_plan method."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '[{"action": "Step 1", "tool": "ExampleTool", "expected_outcome": "Success"}]'
    reasoner = LLMReasoner(mock_llm)

    goal = "Test Goal"
    context = {"key": "value"}

    plan = reasoner.generate_plan(goal, context)

    assert len(plan) == 1
    assert plan[0]["action"] == "Step 1"
    assert plan[0]["tool"] == "ExampleTool"
    assert plan[0]["expected_outcome"] == "Success"

def test_agent_memory_record_observation():
    """Test AgentMemory.record_observation method."""
    memory = AgentMemory()

    observation = {"step": "Test Step", "result": {"success": True}}
    memory.record_observation(observation)

    assert len(memory.observations) == 1
    assert memory.observations[0] == observation

def test_agent_memory_record_plan():
    """Test AgentMemory.record_plan method."""
    memory = AgentMemory()

    plan = [{"step": "Test Step", "details": "Some details"}]
    memory.record_plan(plan)

    assert len(memory.past_plans) == 1
    assert memory.past_plans[0]["timestamp"] == 0
    assert memory.past_plans[0]["plan"] == plan

def test_solve_goal():
    """Test GoalOrientedAgent.solve_goal method."""
    mock_llm = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.execute_tool.return_value = {"success": True, "output": "Output"}

    tools = {"ExampleTool": lambda x: x}
    agent = GoalOrientedAgent(mock_llm, tools, name="Test Agent", description="A test agent")
    agent.sandbox = mock_sandbox  # Replace sandbox with the mock

    goal = "Test Goal"
    context = {"key": "value"}
    result = agent.solve_goal(goal, context, max_iterations=1)

    assert result["ExampleTool"] == "Output"
