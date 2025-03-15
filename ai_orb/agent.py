import json
from typing import Any, Dict, List, Callable
from dataclasses import dataclass, field
from ai_orb.sandbox import SecureSandbox, Tool
import ollama

@dataclass
class Think:
    """
    Think module handles reasoning and planning with the LLM.
    """
    llm: ollama  # LLM provides the interface to interact with the LLM
    tools: Dict[str, Callable]  # Dictionary of available tools, including other agents

    def generate_plan(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a step-by-step plan to achieve the goal.

        Args:
            goal (str): The objective to be achieved
            context (Dict[str, Any]): Current context and information

        Returns:
            List of plan steps with details
        """
        # Generate a summary of tool APIs for the LLM
        tool_descriptions = [
            f"Tool: {tool_name}\nAPI Syntax: {tool_func.__doc__}"
            for tool_name, tool_func in self.tools.items()
        ]
        tool_info = "\n\n".join(tool_descriptions)

        prompt = f"""
        Goal: {goal}
        Context: {json.dumps(context)}

        Available Tools:
        {tool_info}

        Generate a detailed, actionable plan to achieve this goal.
        Each plan step should include:
        - A specific action
        - Tool to use and its input
        - Expected output

        Provide the plan as a JSON list of steps.
        """
        try:
            response = self.llm.generate(model="qwen2.5:0.5b", prompt=prompt)
            return json.loads(response['response'])
        except Exception as e:
            print(f"Error generating plan: {e}")
            return []


@dataclass
class Act:
    """
    Act module handles the execution of tasks in a secure sandbox.
    """
    sandbox: SecureSandbox

    def execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool securely in the sandbox.

        Args:
            tool_name (str): The name of the tool to execute
            input_data (Dict[str, Any]): Input for the tool

        Returns:
            Dict[str, Any]: Result of the tool execution
        """
        try:
            result = self.sandbox.execute_tool(tool_name, input_data)
            return {"success": True, "output": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


@dataclass
class Observe:
    """
    Observe module manages memory and observations.
    """
    observations: List[Dict[str, Any]] = field(default_factory=list)
    past_plans: List[Dict[str, Any]] = field(default_factory=list)

    def record_observation(self, observation: Dict[str, Any]):
        """
        Record an observation from an action.

        Args:
            observation (Dict[str, Any]): Observation details
        """
        self.observations.append(observation)

    def record_plan(self, plan: List[Dict[str, Any]]):
        """
        Record a plan for future reference.

        Args:
            plan (List[Dict[str, Any]]): Plan steps
        """
        self.past_plans.append({
            "timestamp": len(self.past_plans),
            "plan": plan
        })


class CollaborativeAgent:
    """
    A collaborative agent that follows the Think-Act-Observe cycle and collaborates with other agents.
    """
    def __init__(self, llm: ollama, tools: Dict[str, Callable], name: str, description: str):
        """
        Initialize the agent with modules for thinking, acting, and observing.

        Args:
            llm (Tool): Language model for reasoning
            tools (Dict[str, Callable]): Tools for action, including other agents
            name (str): Name of the agent
            description (str): Description of the agent
        """
        sandbox = SecureSandbox(tools)
        self.think = Think(llm, tools)
        self.act = Act(sandbox)
        self.observe = Observe()
        self.name = name
        self.description = description

    def solve_goal(self, goal: str, initial_context: Dict[str, Any], max_iterations: int = 5):
        """
        Solve a goal iteratively through Think-Act-Observe cycle.

        Args:
            goal (str): The objective to achieve
            initial_context (Dict[str, Any]): Starting context
            max_iterations (int): Maximum number of iterations

        Returns:
            Final context after attempting to solve the goal
        """
        current_context = initial_context.copy()

        for iteration in range(max_iterations):
            print(f"[{self.name}] Iteration {iteration + 1}: Thinking...")
            # THINK: Generate plan
            plan = self.think.generate_plan(goal, current_context)
            self.observe.record_plan(plan)

            print(f"[{self.name}] Plan: {plan}")

            # ACT: Execute steps
            for step in plan:
                tool_name = step.get("tool")
                input_data = step.get("input", {})

                print(f"[{self.name}] Executing step: {step}")
                result = self.act.execute_tool(tool_name, input_data)

                # OBSERVE: Record observation
                observation = {"step": step, "result": result}
                self.observe.record_observation(observation)
                print(f"[{self.name}] Observation: {observation}")

                # Update context with result
                if result.get("success"):
                    current_context[tool_name] = result["output"]

                # Check if goal is achieved
                if self._check_goal_achieved(goal, current_context):
                    print(f"[{self.name}] Goal achieved!")
                    return current_context

            # Refine context for the next iteration
            current_context = self._refine_context(current_context)
            print(f"[{self.name}] Refined Context: {current_context}")

        print(f"[{self.name}] Failed to achieve goal within maximum iterations.")
        return current_context

    def _check_goal_achieved(self, goal: str, context: Dict[str, Any]) -> bool:
        """
        Check if the goal has been achieved.

        Args:
            goal (str): The goal
            context (Dict[str, Any]): Current context

        Returns:
            bool: True if goal is achieved, False otherwise
        """
        # Implement goal-specific logic
        return False

    def _refine_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine the context based on observations.

        Args:
            context (Dict[str, Any]): Current context

        Returns:
            Refined context
        """
        # Implement logic for context refinement
        return context


if __name__ == "__main__":
    # Define two agents that can collaborate
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

    tools_agent_1 = {"sample_tool_agent_1": sample_tool_agent_1}
    tools_agent_2 = {"sample_tool_agent_2": sample_tool_agent_2}

    import ollama
    llm = ollama.Client(host='http://localhost:11434')

    agent_1 = CollaborativeAgent(llm, tools_agent_2, "Agent 1", "Collaborative agent 1")
    agent_2 = CollaborativeAgent(llm, tools_agent_1, "Agent 2", "Collaborative agent 2")

    goal = "Complete a complex task collaboratively"
    initial_context = {"sample_tool_agent_1": None, "sample_tool_agent_2": None}

    # Start collaboration with Agent 1 initiating the process
    result = agent_1.solve_goal(goal, initial_context)
    print(f"Final Result from Agent 1: {result}")

    # Agent 2 can further refine or take over based on context
    result = agent_2.solve_goal(goal, result)
    print(f"Final Result from Agent 2: {result}")
