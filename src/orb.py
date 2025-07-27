# © 2025 Charudatta Korde · Licensed under CC BY-NC-SA 4.0 · View License @ https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
# C:\Users\korde\Home\Github\task-runner-SDLC\src\templates/LICENSE
from src.agent import Agent, MCPServer, MCPClient, MCPHost

class AgentUIProtocol:
    """Example UI protocol for agent interaction."""
    def send(self, message):
        # Implement UI send logic (e.g., websocket, REST, etc.)
        pass
    def receive(self):
        # Implement UI receive logic
        pass

def main():
    # Example: create agents and register UI protocol
    agent_ui_protocol = AgentUIProtocol()
    agent = Agent(
        llm=None,  # Replace with actual LLM client
        tools={},
        name="DemoAgent",
        description="Demo agent with UI protocol",
        ui_protocol=agent_ui_protocol
    )
    # Example: send/receive via UI protocol
    agent.send_to_ui({"msg": "Hello UI"})

if __name__ == "__main__":
    main()