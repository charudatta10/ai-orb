class MockAgent:
    def __init__(self, name, success=True):
        self.name = name
        self.success = success

    def solve_goal(self, goal, initial_context):
        if self.success:
            return {f"{self.name}_goal": f"Solved {goal}"}
        else:
            raise Exception("Agent failed to solve the goal.")

import pytest

def test_validate_agents():
    agents = [MockAgent("Agent1"), MockAgent("Agent2")]
    runner = AgentRunner(agents)
    validated_agents = runner._validate_agents(agents)
    assert len(validated_agents) == 2
    assert validated_agents[0].name == "Agent1"
    assert validated_agents[1].name == "Agent2"


def test_aggregate_results():
    runner = AgentRunner([])
    current_context = {"main_goal": "Test goal"}
    agent_results = [
        {"agent_type": "MockAgent1", "result": {"key1": "value1"}},
        {"agent_type": "MockAgent2", "result": {"key2": "value2"}}
    ]
    aggregated_context = runner._aggregate_results(current_context, agent_results)
    assert aggregated_context["key1"] == "value1"
    assert aggregated_context["key2"] == "value2"
    assert aggregated_context["main_goal"] == "Test goal"

def test_check_goal_completion():
    agents = [MockAgent("Agent1"), MockAgent("Agent2")]
    runner = AgentRunner(agents)
    context = {"Agent1_output": "Solved", "Agent2_output": "Solved"}
    assert runner._check_goal_completion(context) is True
    context = {"Agent1_output": "Solved", "Agent2_output": "Failed"}
    assert runner._check_goal_completion(context) is False

def test_run_agents():
    ...

def test_run_agents_with_failure():
    ...

def test_run_agents_with_exception():
    ...

def test_run_agents_with_timeout():
    ...

def test_run_agents_with_invalid_agents():
    ...

def test_run_agents_with_no_agents():
    ...

def test_run_agents_with_no_goal():
    ...

def test_run_agents_with_no_context():
    ...


@pytest.mark.asyncio
async def test_shared_memory():
    memory = AgentSharedMemory()
    await memory.store("key1", "value1")
    assert await memory.retrieve("key1") == "value1"
    await memory.clear()
    assert await memory.retrieve("key1") is None



