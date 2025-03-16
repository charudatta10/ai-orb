import json
from typing import Any, Dict, List, Callable, Optional, Union
from dataclasses import dataclass, field
from ai_orb.sandbox import SecureSandbox, Tool
import ollama
import logging
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class Think:
    """Think module handles reasoning and planning with the LLM."""
    llm: ollama
    tools: Dict[str, Callable]
    model_name: str = "qwen2.5:0.5b"  # Default model, configurable

    def generate_plan(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a step-by-step plan to achieve the goal.

        Args:
            goal: The objective to be achieved
            context: Current context and information

        Returns:
            List of plan steps with details
        """
        # Generate tool descriptions once
        tool_info = self._get_tool_descriptions()
        
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
            response = self.llm.generate(model=self.model_name, prompt=prompt)
            logger.debug(f"Plan generation response: {response}")
            
            if 'response' in response:
                return self._extract_json_snippets(response['response'])
            else:
                logger.error("Unexpected response format from LLM")
                return []
                
        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            return []

    def _get_tool_descriptions(self) -> str:
        """Generate a summary of tool APIs for the LLM."""
        tool_descriptions = [
            f"Tool: {tool_name}\nAPI Syntax: {tool_func.__doc__}"
            for tool_name, tool_func in self.tools.items()
        ]
        return "\n\n".join(tool_descriptions)

    def _extract_json_snippets(self, markdown: str) -> List[Dict[str, Any]]:
        """
        Extract JSON snippets from markdown text.

        Args:
            markdown: Markdown text containing JSON code snippets

        Returns:
            Parsed JSON objects
        """
        # Look for JSON in code blocks with or without json tag
        patterns = [
            r"```json\s*(.*?)\s*```",  # With json tag
            r"```\s*([\[\{].*?[\]\}])\s*```"  # Without json tag but starts with [ or {
        ]
        
        json_blocks = []
        for pattern in patterns:
            json_blocks.extend(re.findall(pattern, markdown, re.DOTALL))
            if json_blocks:  # If we found matches, no need to try other patterns
                break
                
        plans = []
        for block in json_blocks:
            try:
                plan_data = json.loads(block.strip())
                # Handle both single dict and list of dicts
                if isinstance(plan_data, list):
                    plans.extend(plan_data)
                else:
                    plans.append(plan_data)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON block: {e}")
                logger.debug(f"Problematic JSON block: {block}")

        return plans


@dataclass
class Act:
    """Act module handles the execution of tasks in a secure sandbox."""
    sandbox: SecureSandbox

    def execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool securely in the sandbox.

        Args:
            tool_name: The name of the tool to execute
            input_data: Input for the tool

        Returns:
            Result of the tool execution
        """
        try:
            result = self.sandbox.execute(tool_name, input_data)
            return {"success": True, "output": result}
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return {"success": False, "error": str(e)}


@dataclass
class Observe:
    """Observe module manages memory and observations."""
    # Using field(default_factory=list) to avoid mutable default argument problems
    observations: List[Dict[str, Any]] = field(default_factory=list)
    past_plans: List[Dict[str, Any]] = field(default_factory=list)

    def record_observation(self, observation: Dict[str, Any]) -> None:
        """
        Record an observation from an action.

        Args:
            observation: Observation details
        """
        observation_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            **observation
        }
        self.observations.append(observation_with_timestamp)

    def record_plan(self, plan: List[Dict[str, Any]]) -> None:
        """
        Record a plan for future reference.

        Args:
            plan: Plan steps
        """
        self.past_plans.append({
            "timestamp": datetime.now().isoformat(),
            "plan": plan
        })
        
    def get_recent_observations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent observations.
        
        Args:
            limit: Maximum number of observations to return
            
        Returns:
            Recent observations
        """
        return self.observations[-limit:] if self.observations else []


class CollaborativeAgent:
    """A collaborative agent that follows the Think-Act-Observe cycle."""
    
    def __init__(
        self, 
        llm: ollama, 
        tools: Dict[str, Callable], 
        name: str, 
        description: str,
        model_name: str = "qwen2.5:0.5b"
    ):
        """
        Initialize the agent with modules for thinking, acting, and observing.

        Args:
            llm: Language model for reasoning
            tools: Tools for action, including other agents
            name: Name of the agent
            description: Description of the agent
            model_name: Name of the model to use
        """
        sandbox = SecureSandbox(tools)
        self.think = Think(llm, tools, model_name)
        self.act = Act(sandbox)
        self.observe = Observe()
        self.name = name
        self.description = description
        
    def _log_step(self, message: str) -> None:
        """Centralized logging helper."""
        logger.info(f"[{self.name}] {message}")

    def solve_goal(
        self, 
        goal: str, 
        initial_context: Dict[str, Any], 
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Solve a goal iteratively through Think-Act-Observe cycle.

        Args:
            goal: The objective to achieve
            initial_context: Starting context
            max_iterations: Maximum number of iterations

        Returns:
            Final context after attempting to solve the goal
        """
        current_context = initial_context.copy()
        
        for iteration in range(max_iterations):
            self._log_step(f"Iteration {iteration + 1}: Thinking...")
            
            # THINK: Generate plan
            plan = self.think.generate_plan(goal, current_context)
            if not plan:
                self._log_step("Failed to generate a valid plan")
                continue
                
            self.observe.record_plan(plan)
            self._log_step(f"Plan generated with {len(plan)} steps")

            # Execute plan and update context
            success = self._execute_plan(plan, current_context)
            
            # Check if goal is achieved
            if self._check_goal_achieved(goal, current_context):
                self._log_step("Goal achieved!")
                return current_context
                
            if not success:
                self._log_step("Plan execution failed. Refining context...")
                
            # Refine context for next iteration
            current_context = self._refine_context(current_context)

        self._log_step("Failed to achieve goal within maximum iterations.")
        return current_context
        
    def _execute_plan(self, plan: List[Dict[str, Any]], context: Dict[str, Any]) -> bool:
        """
        Execute all steps in a plan and update the context.
        
        Args:
            plan: The plan to execute
            context: The current context to update
            
        Returns:
            True if all steps executed successfully, False otherwise
        """
        all_steps_successful = True
        
        for step_num, step in enumerate(plan, 1):
            tool_name = step.get("tool")
            input_data = step.get("input", {})
            
            if not tool_name:
                self._log_step(f"Step {step_num}: Missing tool name in step")
                all_steps_successful = False
                continue

            self._log_step(f"Step {step_num}: Executing {tool_name}")
            result = self.act.execute_tool(tool_name, input_data)
            
            # Record observation
            observation = {"step_num": step_num, "step": step, "result": result}
            self.observe.record_observation(observation)
            
            # Update context with result or error
            if result.get("success"):
                context[f"result_{tool_name}"] = result["output"]
                self._log_step(f"Step {step_num}: Success")
            else:
                context[f"error_{tool_name}"] = result.get("error", "Unknown error")
                self._log_step(f"Step {step_num}: Failed - {result.get('error')}")
                all_steps_successful = False
        
        return all_steps_successful

    def _check_goal_achieved(self, goal: str, context: Dict[str, Any]) -> bool:
        """
        Check if the goal has been achieved.

        Args:
            goal: The goal
            context: Current context

        Returns:
            True if goal is achieved, False otherwise
        """
        # This would need to be implemented with specific logic
        # For now, we'll check if there are no error entries in the context
        return not any(key.startswith("error_") for key in context.keys())

    def _refine_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine the context based on observations.

        Args:
            context: Current context

        Returns:
            Refined context
        """
        refined_context = context.copy()
        
        # Add recent observations to context
        recent_observations = self.observe.get_recent_observations()
        refined_context["recent_observations"] = recent_observations
        
        # Clear any "None" values as they're not useful
        refined_context = {k: v for k, v in refined_context.items() if v is not None}
        
        return refined_context


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
    agent_1 = CollaborativeAgent(llm, tools_agent_2, "Agent1", "Collaborative agent 1")
    agent_2 = CollaborativeAgent(llm, tools_agent_1, "Agent2", "Collaborative agent 2")

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