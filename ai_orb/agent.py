import json
from typing import Any, Dict, List
from dataclasses import dataclass, field
from tool import Tool
from sandbox import SecureSandbox

@dataclass
class Think:
    """
    Think module handles reasoning and planning with the LLM.
    """
    llm: Tool  # LLMTool provides the interface to interact with the LLM
    tools: Dict[str, callable]  # Dictionary of available tools with their API syntax

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


class GoalOrientedAgent:
    """
    An agent following the Think-Act-Observe cycle to achieve a goal.
    """
    def __init__(self, llm: Tool, tools: Dict[str, callable], name: str, description: str):
        """
        Initialize the agent with modules for thinking, acting, and observing.

        Args:
            llm (LLMTool): Language model for reasoning
            tools (Dict[str, callable]): Tools for action
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
            print(f"Iteration {iteration + 1}: Thinking...")
            # THINK: Generate plan
            plan = self.think.generate_plan(goal, current_context)
            self.observe.record_plan(plan)

            print(f"Plan: {plan}")

            # ACT: Execute steps
            for step in plan:
                tool_name = step.get("tool")
                input_data = step.get("input", {})

                print(f"Executing step: {step}")
                result = self.act.execute_tool(tool_name, input_data)

                # OBSERVE: Record observation
                observation = {"step": step, "result": result}
                self.observe.record_observation(observation)
                print(f"Observation: {observation}")

                # Update context with result
                if result.get("success"):
                    current_context[tool_name] = result["output"]

                # Check if goal is achieved
                if self._check_goal_achieved(goal, current_context):
                    print("Goal achieved!")
                    return current_context

            # Refine context for the next iteration
            current_context = self._refine_context(current_context)
            print(f"Refined Context: {current_context}")

        print("Failed to achieve goal within maximum iterations.")
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
    # Example usage with LLMTool and SecureSandbox
    def sample_tool(input_data):
        """
        API Syntax: sample_tool(input_data: Dict[str, Any]) -> str
        Processes the input and returns the result.
        """
        return f"Processed {input_data}"

    tools = {"sample_tool": sample_tool}
    import ollama
    llm = ollama.Client(host='http://localhost:11434')
    agent = GoalOrientedAgent(llm, tools, "Test Agent", "An agent for testing.")
    goal = "Complete a sample task"
    initial_context = {"sample_tool": None}
    result = agent.solve_goal(goal, initial_context)
    print(f"Final Result: {result}")
