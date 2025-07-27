# © 2025 Charudatta Korde · Licensed under CC BY-NC-SA 4.0 · View License @ https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
# C:\Users\korde\Home\Github\task-runner-SDLC\src\templates/LICENSE
from src.agent import Agent, MCPServer, MCPHost, MCPClient
import ollama

def sample_tool_agent_1(input_data):
    """API Syntax: sample_tool_agent_1(input_data: Dict[str, Any]) -> str"""
    return f"Agent 1 processed {input_data}"

def sample_tool_agent_2(input_data):
    """API Syntax: sample_tool_agent_2(input_data: Dict[str, Any]) -> str"""
    return f"Agent 2 processed {input_data}"

def main():
    # Define tools for each agent
    tools_agent_1 = {"sample_tool_agent_1": sample_tool_agent_1}
    tools_agent_2 = {"sample_tool_agent_2": sample_tool_agent_2}

    # Connect to Ollama
    llm = ollama.Client(host='http://localhost:11434')

    # Create agents (with protocol stubs for demonstration)
    agent_1 = Agent(llm, tools_agent_2, "Agent1", "Collaborative agent 1")
    agent_2 = Agent(llm, tools_agent_1, "Agent2", "Collaborative agent 2")

    # Register agents with MCP server/host
    mcp_server = MCPServer([agent_1, agent_2])
    mcp_host = MCPHost([agent_1, agent_2])

    # Example: orchestrate agents via MCP host (stub)
    mcp_host.orchestrate()

    # Example: agent solves a goal
    goal = "use both agents to process data"
    initial_context = {"data": "Example data for processing"}
    result_1 = agent_1.solve_goal(goal, initial_context)
    print(f"Final Result from Agent 1: {result_1}")

    result_2 = agent_2.solve_goal(goal, result_1)
    print(f"Final Result from Agent 2: {result_2}")

    return result_2

if __name__ == "__main__":
    main()