import json
from typing import Any, Dict, List, Callable, Optional, Union
from dataclasses import dataclass, field
from ai_orb.sandbox import SecureSandbox, Tool
from ai_orb.think import Think
from ai_orb.act import Act
from ai_orb.observe import Observe
import ollama
import logging
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Agent:
    """A collaborative agent that follows the Think-Act-Observe cycle."""
    
    def __init__(
        self, 
        llm: Any, 
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





if __name__ == "__main__":
    ...