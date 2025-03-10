from agent import GoalOrientedAgent
from runner import AgentRunner
from tool import LLMTool
import ollama
import asyncio
import json
from sandbox import SecureSandbox

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
        "summarize": LLMTool(
            summarize, 
            name="summarize", 
            description="Summarize a given text"
        ),
        "classify": LLMTool(
            classify, 
            name="classify", 
            description="Classify a given text"
        ),
        "generate": LLMTool(
            generate, 
            name="generate", 
            description="Generate text based on a prompt"
        ),
        "translate": LLMTool(
            translate, 
            name="translate", 
            description="Translate text to a target language"
        ),
        "qa": LLMTool(
            qa, 
            name="qa", 
            description="Answer a question based on a given context"
        )
    }
    
    # Create properly structured multi-agent system
    agents = {
        "research_agent": GoalOrientedAgent(
            llm.chat(model=model_name), 
            sandbox,
            tools,
            name="ResearchAgent",
            description="Researches current AI trends"
        ),
        "analysis_agent": GoalOrientedAgent(
            llm.chat(model=model_name), 
            sandbox,
            tools,
            name="AnalysisAgent",
            description="Analyzes and synthesizes AI trend information"
        )
    }
    
    # Initialize runner with proper agents
    runner = AgentRunner(
        agents=agents,
        problem_context={"initial_directive": "Investigate AI trends"}
    )
    
    try:
        # Run collaborative problem solving with error handling
        result = await runner.run_collaborative_solve(
            main_goal="Generate comprehensive AI trend report",
            max_iterations=3
        )
        
        print("Final Result:", json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        # Fallback logic in case of failure
        print("Attempting basic report generation without agents...")
        basic_report = {
            "title": "AI Trends Report",
            "sections": [
                {"heading": "Current Trends", "content": "AI adoption is increasing across industries."},
                {"heading": "Future Outlook", "content": "Continued growth expected in generative AI applications."}
            ],
            "generated_by": "Fallback mechanism"
        }
        print(json.dumps(basic_report, indent=2))

if __name__ == "__main__":
    asyncio.run(main())