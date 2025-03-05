import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import traceback

class LLMReasoner:
    """
    Provides reasoning capabilities using a Language Model.
    """
    def __init__(self, llm):
        """
        Initialize the reasoner with a language model.
        
        Args:
            llm: Language model with generate or chat method
        """
        self.llm = llm
    
    def generate_plan(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a step-by-step plan to achieve the goal.
        
        Args:
            goal (str): The objective to be achieved
            context (Dict[str, Any]): Current context and available information
        
        Returns:
            List of plan steps with details
        """
        prompt = f"""
        Goal: {goal}
        Context: {json.dumps(context)}
        
        Generate a detailed, actionable plan to achieve this goal. 
        Each plan step should include:
        - A clear, specific action
        - Required tools or resources
        - Expected outcome
        
        Provide the plan as a JSON list of steps.
        """
        
        try:
            response = self.llm.generate(prompt)
            return json.loads(response)
        except Exception as e:
            print(f"Error generating plan: {e}")
            return []

class ToolSandbox:
    """
    Provides a safe execution environment for tools.
    """
    def __init__(self, tools: Dict[str, callable]):
        """
        Initialize the sandbox with available tools.
        
        Args:
            tools (Dict[str, callable]): Dictionary of available tools
        """
        self.tools = tools
    
    def execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool in a controlled environment.
        
        Args:
            tool_name (str): Name of the tool to execute
            input_data (Dict[str, Any]): Input parameters for the tool
        
        Returns:
            Execution result or error information
        """
        try:
            tool = self.tools.get(tool_name)
            if not tool:
                return {"error": f"Tool {tool_name} not found"}
            
            result = tool(**input_data)
            return {
                "success": True,
                "output": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

@dataclass
class AgentMemory:
    """
    Manages agent's memory and learning.
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
    An agent that can reason, plan, act, and learn towards achieving a goal.
    """
    def __init__(self, llm, tools: Dict[str, callable]):
        """
        Initialize the agent with reasoning and execution capabilities.
        
        Args:
            llm: Language model for reasoning
            tools (Dict[str, callable]): Available tools for action
        """
        self.reasoner = LLMReasoner(llm)
        self.sandbox = ToolSandbox(tools)
        self.memory = AgentMemory()
    
    def solve_goal(self, goal: str, initial_context: Dict[str, Any], max_iterations: int = 5):
        """
        Attempt to solve a goal through iterative planning and execution.
        
        Args:
            goal (str): The objective to achieve
            initial_context (Dict[str, Any]): Starting context and information
            max_iterations (int): Maximum number of planning iterations
        
        Returns:
            Final result of goal-solving attempt
        """
        current_context = initial_context.copy()
        
        for iteration in range(max_iterations):
            # Generate plan
            plan = self.reasoner.generate_plan(goal, current_context)
            self.memory.record_plan(plan)
            
            # Execute plan steps
            for step in plan:
                tool_name = step.get('tool')
                input_data = step.get('input', {})
                
                # Execute tool in sandbox
                result = self.sandbox.execute_tool(tool_name, input_data)
                
                # Record observation
                observation = {
                    "step": step,
                    "result": result
                }
                self.memory.record_observation(observation)
                
                # Update context with result
                if result.get('success'):
                    current_context[tool_name] = result['output']
                
                # Optional: Check goal achievement
                if self._check_goal_achieved(goal, current_context):
                    return current_context
            
            # If goal not achieved, use memory to refine approach
            current_context = self._refine_context(current_context)
        
        return current_context
    
    def _check_goal_achieved(self, goal: str, context: Dict[str, Any]) -> bool:
        """
        Check if the goal has been achieved based on current context.
        
        Args:
            goal (str): Goal to evaluate
            context (Dict[str, Any]): Current context
        
        Returns:
            Boolean indicating goal achievement
        """
        # Implement goal-specific achievement logic
        # This is a placeholder and should be customized
        return False
    
    def _refine_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine context based on past observations and failures.
        
        Args:
            context (Dict[str, Any]): Current context
        
        Returns:
            Refined context
        """
        # Implement context refinement logic
        # This could involve LLM-based analysis of past observations
        return context

# Example usage
def example_tools():
    """
    Create example tools for demonstration.
    """
    def search_internet(query):
        # Simulated internet search
        return f"Search results for: {query}"
    
    def analyze_data(data):
        # Simulated data analysis
        return f"Analysis of {data}"
    
    def generate_report(data):
        # Simulated report generation
        return f"Report generated from {data}"
    
    return {
        "search": search_internet,
        "analyze": analyze_data,
        "report": generate_report
    }

# Simulation of a simple LLM
class SimpleLLM:
    def generate(self, prompt):
        # Very basic LLM simulation
        return json.dumps([
            {
                "tool": "search",
                "input": {"query": "research problem"},
                "description": "Perform initial internet search"
            },
            {
                "tool": "analyze",
                "input": {"data": "search results"},
                "description": "Analyze gathered information"
            },
            {
                "tool": "report",
                "input": {"data": "analysis results"},
                "description": "Generate final report"
            }
        ])

# Demonstration
def main():
    # Create tools and LLM
    tools = example_tools()
    llm = SimpleLLM()
    
    # Initialize agent
    agent = GoalOrientedAgent(llm, tools)
    
    # Solve a goal
    result = agent.solve_goal(
        goal="Generate a comprehensive research report",
        initial_context={"initial_query": "AI advancements"}
    )
    
    print("Final Context:", result)
    print("Agent Memory:", agent.memory.observations)

if __name__ == "__main__":
    main()