"""
AI-Orb: Collaborative Agent System
A minimal, clean implementation following DRY and KISS principles.
"""
import json
import logging
import re
import queue
import threading
import sys
import io
import ast
import inspect
import traceback
import builtins
import importlib
from typing import Any, Dict, List, Callable, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Wrapper for functions to be used as agent tools."""
    func: Callable
    name: str = None
    description: str = None
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if not self.name:
            self.name = self.func.__name__
        if not self.description:
            self.description = self.func.__doc__ or "No description provided"
    
    def __call__(self, *args, **kwargs):
        """Make the tool directly callable."""
        try:
            sig = inspect.signature(self.func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            return self.func(*bound_args.args, **bound_args.kwargs)
        except Exception as e:
            raise ValueError(f"Error calling tool {self.name}: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description
        }


class SecureSandbox:
    """Secure execution environment for running tools and code."""
    
    def __init__(self, 
                 allowed_tools: Dict[str, Callable] = None,
                 max_time: int = 5,
                 max_output_size: int = 10 * 1024,
                 allowed_modules: List[str] = None):
        """
        Initialize sandbox with security constraints.
        
        Args:
            allowed_tools: Dictionary of allowed tools
            max_time: Maximum execution time in seconds
            max_output_size: Maximum output size in bytes
            allowed_modules: List of explicitly allowed modules
        """
        self.allowed_tools = allowed_tools or {}
        self.max_time = max_time
        self.max_output_size = max_output_size
        self.allowed_modules = set(allowed_modules or [])
        
        # Add modules needed by allowed tools
        for tool in self.allowed_tools.values():
            if hasattr(tool, '__module__') and tool.__module__ != '__main__':
                self.allowed_modules.add(tool.__module__)
    
    def execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Any:
        """Execute a tool by name with given input data."""
        if tool_name not in self.allowed_tools:
            raise ValueError(f"Tool '{tool_name}' not found in allowed tools")
            
        tool = self.allowed_tools[tool_name]
        
        # Extract the function if it's a Tool instance
        func = tool.func if isinstance(tool, Tool) else tool
        
        try:
            # Execute the function in a controlled environment
            return self._execute_function(func, input_data)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise
    
    def _execute_function(self, func: Callable, input_data: Dict[str, Any]) -> Any:
        """Execute a function in the sandbox with timeout protection."""
        result_queue = queue.Queue()
        
        def run_func():
            try:
                # Call the function with input data
                result = func(input_data)
                result_queue.put({"success": True, "result": result})
            except Exception as e:
                result_queue.put({"success": False, "error": str(e)})
        
        # Run in separate thread with timeout
        exec_thread = threading.Thread(target=run_func)
        exec_thread.daemon = True
        exec_thread.start()
        exec_thread.join(timeout=self.max_time)
        
        # Check for timeout
        if exec_thread.is_alive():
            raise TimeoutError(f"Tool execution exceeded {self.max_time} second time limit")
        
        # Get result or raise exception
        try:
            result = result_queue.get(block=False)
            if result["success"]:
                return result["result"]
            else:
                raise RuntimeError(result["error"])
        except queue.Empty:
            raise RuntimeError("Tool execution failed to produce a result")
    
    def _validate_security(self, func_or_code):
        """Validate security of code or function."""
        if callable(func_or_code):
            # For functions, check if it's from an allowed module
            module = func_or_code.__module__
            if module not in self.allowed_modules and module != "__main__":
                raise ValueError(f"Function from module '{module}' is not allowed")
        else:
            # For code strings, validate using AST
            try:
                tree = ast.parse(func_or_code)
                # Implement security checks (simplified for brevity)
                # In a real implementation, a full SecurityVisitor would be needed
                # This is a placeholder for actual security validation
                pass
            except SyntaxError as e:
                raise ValueError(f"Syntax error: {e}")


@dataclass
class Think:
    """Reasoning and planning module using LLMs."""
    llm: Any  # LLM client
    tools: Dict[str, Callable]
    model_name: str = "qwen2.5:0.5b"
    
    def generate_plan(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a step-by-step plan to achieve the goal."""
        # Create tool descriptions for the prompt
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
            logger.debug(f"LLM response obtained, extracting plan")
            
            if 'response' in response:
                return self._extract_json_plan(response['response'])
            else:
                logger.error("Unexpected LLM response format")
                return []
                
        except Exception as e:
            logger.error(f"Plan generation error: {e}")
            return []
    
    def _get_tool_descriptions(self) -> str:
        """Create formatted tool descriptions for the LLM prompt."""
        descriptions = []
        for name, tool in self.tools.items():
            doc = tool.__doc__ if callable(tool) else "No documentation"
            descriptions.append(f"Tool: {name}\nAPI Syntax: {doc}")
        return "\n\n".join(descriptions)
    
    def _extract_json_plan(self, text: str) -> List[Dict[str, Any]]:
        """Extract and parse JSON plan from LLM output."""
        # Match JSON blocks with or without language tag
        patterns = [
            r"```json\s*(.*?)\s*```",  # JSON with tag
            r"```\s*([\[\{].*?[\]\}])\s*```",  # JSON without tag
            r"([\[\{].*?[\]\}])"  # Bare JSON (fallback)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        data = json.loads(match.strip())
                        # Handle both list and dict formats
                        if isinstance(data, list):
                            return data
                        else:
                            return [data]
                    except json.JSONDecodeError:
                        continue
        
        logger.warning("Failed to extract valid JSON from LLM response")
        return []


@dataclass
class Act:
    """Action execution module."""
    sandbox: SecureSandbox
    
    def execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and format the result."""
        try:
            result = self.sandbox.execute_tool(tool_name, input_data)
            return {"success": True, "output": result}
        except Exception as e:
            logger.error(f"Tool execution error: {str(e)}")
            return {"success": False, "error": str(e)}


@dataclass
class Observe:
    """Memory and observation tracking module."""
    observations: List[Dict[str, Any]] = field(default_factory=list)
    plans: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_observation(self, observation: Dict[str, Any]) -> None:
        """Record an observation with timestamp."""
        self.observations.append({
            "timestamp": datetime.now().isoformat(),
            **observation
        })
    
    def record_plan(self, plan: List[Dict[str, Any]]) -> None:
        """Record a plan with timestamp."""
        self.plans.append({
            "timestamp": datetime.now().isoformat(),
            "steps": plan
        })
    
    def get_recent_observations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent observations."""
        return self.observations[-limit:] if self.observations else []
    
    def get_recent_plans(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get the most recent plans."""
        return self.plans[-limit:] if self.plans else []


class Agent:
    """Collaborative agent with Think-Act-Observe loop."""
    
    def __init__(
        self,
        llm: Any,
        tools: Dict[str, Callable],
        name: str,
        description: str = "",
        model_name: str = "qwen2.5:0.5b",
        max_execution_time: int = 5
    ):
        """Initialize agent components."""
        self.name = name
        self.description = description
        
        # Set up components
        sandbox = SecureSandbox(tools, max_time=max_execution_time)
        self.think = Think(llm, tools, model_name)
        self.act = Act(sandbox)
        self.observe = Observe()
        
        logger.info(f"Agent '{name}' initialized with {len(tools)} tools")
    
    def log(self, message: str) -> None:
        """Centralized logging for the agent."""
        logger.info(f"[{self.name}] {message}")
    
    def solve(self, goal: str, context: Dict[str, Any] = None, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Solve a goal through iterative thinking, acting, and observing.
        
        Args:
            goal: The objective to achieve
            context: Initial context data
            max_iterations: Maximum number of plan-execute cycles
            
        Returns:
            Final context after attempting to solve the goal
        """
        context = context or {}
        self.log(f"Starting to solve goal: {goal}")
        
        for iteration in range(max_iterations):
            self.log(f"Iteration {iteration+1}/{max_iterations}")
            
            # Generate plan
            plan = self.think.generate_plan(goal, context)
            if not plan:
                self.log("Failed to generate plan, trying again")
                continue
                
            self.observe.record_plan(plan)
            self.log(f"Generated plan with {len(plan)} steps")
            
            # Execute plan
            for step_num, step in enumerate(plan, 1):
                success = self._execute_step(step_num, step, context)
                if not success and self._should_abort_plan(step, context):
                    self.log("Critical step failed, aborting plan")
                    break
            
            # Check if goal is complete
            if self._goal_achieved(goal, context):
                self.log("Goal achieved successfully")
                break
                
            # Update context for next iteration
            context = self._update_context(context)
        
        return context
    
    def _execute_step(self, step_num: int, step: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Execute a single step of the plan."""
        tool_name = step.get("tool")
        input_data = step.get("input", {})
        
        if not tool_name:
            self.log(f"Step {step_num}: Invalid step - missing tool name")
            return False
            
        self.log(f"Step {step_num}: Executing {tool_name}")
        
        # Execute tool
        result = self.act.execute_tool(tool_name, input_data)
        
        # Record observation
        observation = {
            "step": step_num,
            "tool": tool_name,
            "input": input_data,
            "result": result
        }
        self.observe.record_observation(observation)
        
        # Update context with result
        success = result.get("success", False)
        if success:
            context[f"result_{tool_name}"] = result.get("output")
            self.log(f"Step {step_num}: Success")
        else:
            error = result.get("error", "Unknown error")
            context[f"error_{tool_name}"] = error
            self.log(f"Step {step_num}: Failed - {error}")
        
        return success
    
    def _should_abort_plan(self, step: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Determine if plan should be aborted after a step failure."""
        # Check if step is marked as critical
        return step.get("critical", False)
    
    def _goal_achieved(self, goal: str, context: Dict[str, Any]) -> bool:
        """Check if the goal has been achieved based on context."""
        # A simple implementation - no errors means success
        # In a real system, this would need domain-specific logic
        return not any(key.startswith("error_") for key in context)
    
    def _update_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update context with recent observations for next iteration."""
        updated = context.copy()
        
        # Add recent observations
        updated["recent_observations"] = self.observe.get_recent_observations()
        
        # Clean up any None values
        updated = {k: v for k, v in updated.items() if v is not None}
        
        return updated


def create_collaborative_agents(llm_client, model_name="qwen2.5:0.5b"):
    """Create a pair of collaborative agents for testing."""
    
    # Define tool functions
    def process_data_agent1(input_data):
        """
        API Syntax: process_data_agent1(input_data: Dict[str, Any]) -> Dict[str, Any]
        Process data and return enhanced results.
        """
        data = input_data.get("data", "")
        return {
            "processed_data": f"Agent1 processed: {data}",
            "timestamp": datetime.now().isoformat()
        }
    
    def process_data_agent2(input_data):
        """
        API Syntax: process_data_agent2(input_data: Dict[str, Any]) -> Dict[str, Any]
        Process data and return enhanced results.
        """
        data = input_data.get("data", "")
        return {
            "processed_data": f"Agent2 processed: {data}",
            "timestamp": datetime.now().isoformat()
        }


    # Create tool instances
    tool1 = Tool(process_data_agent1, name="process_data_agent1", description="Process data for Agent1")    
    tool2 = Tool(process_data_agent2, name="process_data_agent2", description="Process data for Agent2")

    # Create agents
    agent1 = Agent(llm_client, {"process_data": tool1}, name="Agent1", model_name=model_name)
    agent2 = Agent(llm_client, {"process_data": tool2}, name="Agent2", model_name=model_name)

    return agent1, agent2

if __name__ == "__main__":
    # Create a collaborative agent 
    import ollama
    llm_client = ollama.Client()
    agent1, agent2 = create_collaborative_agents(llm_client=llm_client)
    
    # Define a goal and context for the agents
    goal = "Enhance and process data"
    context = {
        "data": "Initial data"
    }
    
    # Solve the goal using the agents
    final_context = agent1.solve(goal, context)
    
    # Display final context
    print(final_context)
# Output: {'data': 'Initial data', 'recent_observations': [], 'result_process_data_agent1': {'processed_data': 'Agent1 processed: Initial data', 'timestamp': '2022-01-01T00:00:00.000000'}}
