from ai_orb.agent import CollaborativeAgent, SecureSandbox, Tool
import ollama
import asyncio
import json
from typing import Any, Callable, Dict, Optional
import inspect

# Define the tools available to the agent as actual functions
def summarize(text):
    # Implement the summarization logic here
    return f"Summary of: {text[:50]}..."

def classify(text):
    # Implement the classification logic here
    return "Classification: informational"

def generate(prompt):
    # Implement the text generation logic here
    return f"Generated content based on: {prompt}"

def translate(text, target_language):
    # Implement the translation logic here
    return f"Translated to {target_language}: {text[:30]}..."

def qa(question, context):
    # Implement the question answering logic here
    return f"Answer to '{question}' based on context"


#  Tool Example usage -------------------------------------------------------------------------------------------
def example_tool(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example tool function for demonstration.
        
        This function simply returns the input data as-is.
        
        Args:
            input_data (Dict[str, Any]): Input data to be returned
        
        Returns:
            Dict[str, Any]: The same input data
        """
        return input_data
    
# Create a tool instance
tool = Tool(example_tool, name="example_tool", description="Simple example tool")
    
# Call the tool directly
result = tool({"key": "value"})
print(result)  # Output: {'key': 'value'}
    
# Convert the tool to a dictionary
tool_dict = tool.to_dict()
print(tool_dict)

#  sandbox Example usage -------------------------------------------------------------------------------------------
sandbox = SecureSandbox(allowed_modules=['math'])
code = """
import math # Allowed module
print(math.sqrt(16))
print("Hello, world!")
"""
result = sandbox.execute(code)
print(result)  # Should print the square root of 16 and "Hello, world!"

def add(a, b):
    return a + b

secure_add = sandbox.execute(add)
result = secure_add(3, 4)
print(result)  # Should print 7

#  CollaborativeAgent Example usage -------------------------------------------------------------------------------------------
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

# Define the main function to demonstrate the multi-agent runner
async def main():
    """
    Demonstrate the multi-agent runner with SecureSandbox.
    """
    # Specify a model that exists in your Ollama installation
    model_name = "qwen2.5:0.5b"  # Update this to a model you have installed
    
    # Simulate language model and tools
    llm = ollama.Client(host='http://localhost:11434')
    
    # Create secure sandbox for tool execution with proper security constraints
    sandbox = SecureSandbox(
        allowed_modules=["json", "re", "string"],
    )
    
    # Wrap tools with secure sandbox
    tools = {
        "summarize": Tool(
            summarize, 
            name="summarize", 
            description="Summarize a given text"
        ),
        "classify": Tool(
            classify, 
            name="classify", 
            description="Classify a given text"
        ),
        "generate": Tool(
            generate, 
            name="generate", 
            description="Generate text based on a prompt"
        ),
        "translate": Tool(
            translate, 
            name="translate", 
            description="Translate text to a target language"
        ),
        "qa": Tool(
            qa, 
            name="qa", 
            description="Answer a question based on a given context"
        )
    }
    
    # Create properly structured multi-agent system
    agents = {
        "agent1": CollaborativeAgent(
            name=model_name,
            tools={
                "summarize": tools["summarize"],
                "classify": tools["classify"],
                "generate": tools["generate"]
            },
            llm=llm,
            description="Agent 1"
        ),
        "agent2": CollaborativeAgent(
            name=model_name,
            tools={
                "translate": tools["translate"],
                "qa": tools["qa"]
            },
            llm=llm,
            description="Agent 2"
        )
    }
    
    # Initialize runner with proper agents
    

if __name__ == "__main__":
    asyncio.run(main())