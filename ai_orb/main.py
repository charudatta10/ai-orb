from ai_orb.agent import Agent, Act, Think, Observe, SecureSandbox, Tool
import ollama


def main():
    """Example usage of collaborative agents."""
    # Define tools for each agent
    def sample_tool_agent_1(input_data):
        """
        API Syntax: sample_tool_agent_1(input_data: Dict[str, Any]) -> str
        Processes the input and returns the result.
        """
        return f"Agent 1 processed {input_data}"

    def sample_tool_agent_2(input_data):
        """
        API Syntax: sample_tool_agent_2(input_data: Dict[str, Any]) -> str
        Processes the input and returns the result.
        """
        return f"Agent 2 processed {input_data}"

    # Define tools for each agent (allow agents to call each other)
    tools_agent_1 = {"sample_tool_agent_1": sample_tool_agent_1}
    tools_agent_2 = {"sample_tool_agent_2": sample_tool_agent_2}

    # Connect to Ollama
    llm = ollama.Client(host='http://localhost:11434')

    # Create agents
    agent_1 = Agent(llm, tools_agent_2, "Agent1", "Collaborative agent 1")
    agent_2 = Agent(llm, tools_agent_1, "Agent2", "Collaborative agent 2")

    # Define goal and initial context
    goal = "use both agents to process data"
    initial_context = {"data": "Example data for processing"}

    # Start collaboration with Agent 1
    result_1 = agent_1.solve_goal(goal, initial_context)
    logger.info(f"Final Result from Agent 1: {result_1}")

    # Agent 2 can further refine
    result_2 = agent_2.solve_goal(goal, result_1)
    logger.info(f"Final Result from Agent 2: {result_2}")
    
    return result_2




if __name__ == "__main__":
    main()