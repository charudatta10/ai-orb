# © 2025 Charudatta Korde · Licensed under CC BY-NC-SA 4.0 · View License @ https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
# C:\Users\korde\Home\Github\task-runner-SDLC\src\templates/LICENSE
import json
from typing import Any, Dict, List, Callable, Optional, Union
from dataclasses import dataclass, field
from src.sandbox import SecureSandbox, Tool
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
    llm: Any  # LLM client
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


if __name__ == "__main__":
    # Example usage
    llm = ollama.Client()
    tools = {
        "tool1": lambda x: x + 1,
        "tool2": lambda x: x * 2
    }
    think = Think(llm=llm, tools=tools)
    goal = "Generate a list of prime numbers"
    context = {"input": 10}
    plan = think.generate_plan(goal, context)
    print(plan)
# Output: [{'tool': 'tool1', 'input': 10}, {'tool': 'tool2', 'input': 11}]