from ai_orb.agent import CollaborativeAgent, SecureSandbox, Tool
import ollama
import asyncio
import json

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