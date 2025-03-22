import markdown
from bs4 import BeautifulSoup
from ollama import chat
from codejail.jail import safe_exec

class Agent:
    def __init__(self, llm_model_name, sandbox_config):
        self.llm_model_name = llm_model_name
        self.sandbox_config = sandbox_config
        self.memory = []

    def process(self, goal, tool):
        plan = self.think(goal, tool)
        tool_call, data = self.parse(plan)
        output = self.act(tool_call, data)
        return self.observe(output)

    def think(self, goal, tool):
        # Generate a plan using the Ollama library
        response = chat(
            model=self.llm_model_name,
            messages=[{'role': 'user', 'content': f"Create a plan to achieve '{goal}' using '{tool}'. Format the plan as Markdown."}]
        )
        return response['message']  # Assuming the response contains a Markdown-formatted plan

    def parse(self, plan):
        # Convert Markdown plan to HTML and parse it to extract the tool and data
        html = markdown.markdown(plan)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract tool and data (assume specific structure in the Markdown)
        tool_call = soup.find('code').text if soup.find('code') else None
        data = soup.find('pre').text if soup.find('pre') else None
        
        return tool_call, data

    def act(self, tool, data):
        # Execute the tool in a sandbox environment using CodeJail
        exec_globals = {}
        code = f"""
def {tool}(data):
    return f"Tool {tool} processed: {{data}}"

result = {tool}("{data}")
"""
        safe_exec(code, exec_globals, **self.sandbox_config)
        return exec_globals.get('result')

    def observe(self, output):
        # Make a decision based on the output and save it to memory
        decision = f"Decision made based on: {output}"
        self.memory.append({'output': output, 'decision': decision})
        return decision


# Example usage
if __name__ == "__main__":
    # Configure sandbox security (example config, customize as needed)
    sandbox_config = {
        'limit_memory': True,
        'limit_cpu_time': True,
        'extra_dirs': [],
    }

    # Instantiate the Agent
    llm_model_name = "llama3.2"  # Replace with the specific Ollama model name
    agent = Agent(llm_model_name, sandbox_config)

    # Run the Agent
    goal = "Analyze trends in customer data"
    tool = "AnalyzeTrends"
    decision = agent.process(goal, tool)

    print(f"Final Decision: {decision}")