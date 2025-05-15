import json
from typing import Any, Dict, List, Callable, Optional, Union, Protocol
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

class AgentProtocol(Protocol):
    """Protocol for agent-to-agent and agent-UI communication."""
    def send(self, message: Dict[str, Any]) -> Any: ...
    def receive(self) -> Dict[str, Any]: ...

class Agent:
    """Base class for collaborative agents with protocol support."""

    def __init__(
        self,
        llm: Any,
        tools: Dict[str, Callable],
        name: str,
        description: str,
        model_name: str = "qwen3:0.6b",
        agent_protocol: Optional[AgentProtocol] = None,
        ui_protocol: Optional[AgentProtocol] = None,
    ):
        self.sandbox = SecureSandbox(tools)
        self.think = Think(llm, tools, model_name)
        self.act = Act(self.sandbox)
        self.observe = Observe()
        self.name = name
        self.description = description
        self.agent_protocol = agent_protocol
        self.ui_protocol = ui_protocol

    def solve_goal(
        self,
        goal: str,
        initial_context: Dict[str, Any],
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        current_context = initial_context.copy()
        for iteration in range(max_iterations):
            logger.info(f"[{self.name}] Iteration {iteration + 1}: Thinking...")
            plan = self.think.generate_plan(goal, current_context)
            if not plan:
                logger.info(f"[{self.name}] Failed to generate a valid plan")
                continue
            self.observe.record_plan(plan)
            logger.info(f"[{self.name}] Plan generated with {len(plan)} steps")
            success = self._execute_plan(plan, current_context)
            if self._check_goal_achieved(goal, current_context):
                logger.info(f"[{self.name}] Goal achieved!")
                return current_context
            if not success:
                logger.info(f"[{self.name}] Plan execution failed. Refining context...")
            current_context = self._refine_context(current_context)
        logger.info(f"[{self.name}] Failed to achieve goal within maximum iterations.")
        return current_context

    def _execute_plan(self, plan: List[Dict[str, Any]], context: Dict[str, Any]) -> bool:
        all_steps_successful = True
        for step_num, step in enumerate(plan, 1):
            tool_name = step.get("tool")
            input_data = step.get("input", {})
            if not tool_name:
                logger.info(f"[{self.name}] Step {step_num}: Missing tool name in step")
                all_steps_successful = False
                continue
            logger.info(f"[{self.name}] Step {step_num}: Executing {tool_name}")
            result = self.act.execute_tool(tool_name, input_data)
            observation = {"step_num": step_num, "step": step, "result": result}
            self.observe.record_observation(observation)
            if result.get("success"):
                context[f"result_{tool_name}"] = result["output"]
                logger.info(f"[{self.name}] Step {step_num}: Success")
            else:
                context[f"error_{tool_name}"] = result.get("error", "Unknown error")
                logger.info(f"[{self.name}] Step {step_num}: Failed - {result.get('error')}")
                all_steps_successful = False
        return all_steps_successful

    def _check_goal_achieved(self, goal: str, context: Dict[str, Any]) -> bool:
        return not any(key.startswith("error_") for key in context.keys())

    def _refine_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        refined_context = context.copy()
        recent_observations = self.observe.get_recent_observations()
        refined_context["recent_observations"] = recent_observations
        refined_context = {k: v for k, v in refined_context.items() if v is not None}
        return refined_context

    # Protocol hooks
    def send_to_agent(self, message: Dict[str, Any]) -> Any:
        if self.agent_protocol:
            return self.agent_protocol.send(message)
        raise NotImplementedError("Agent protocol not set.")

    def receive_from_agent(self) -> Dict[str, Any]:
        if self.agent_protocol:
            return self.agent_protocol.receive()
        raise NotImplementedError("Agent protocol not set.")

    def send_to_ui(self, message: Dict[str, Any]) -> Any:
        if self.ui_protocol:
            return self.ui_protocol.send(message)
        raise NotImplementedError("UI protocol not set.")

    def receive_from_ui(self) -> Dict[str, Any]:
        if self.ui_protocol:
            return self.ui_protocol.receive()
        raise NotImplementedError("UI protocol not set.")

# Placeholder classes for MCP server/client/host
class MCPServer:
    """Multi-Agent Control Protocol Server (stub)."""
    def __init__(self, agents: List[Agent]):
        self.agents = agents
    def start(self):
        # Implement gRPC or SEE protocol server logic here
        # Example: Start both SEE and gRPC servers (stubs)
        logger.info("Starting MCPServer (stub) - gRPC/SEE protocol not implemented.")

        # SEE Protocol stub
        def see_server():
            logger.info("SEE protocol server started (stub).")
            # Here you would implement the SEE protocol server logic
            # For example, listen for incoming messages and route to agents

        # gRPC Protocol stub
        def grpc_server():
            logger.info("gRPC protocol server started (stub).")
            # Here you would implement the gRPC server logic
            # For example, define protobufs, start gRPC server, handle requests

        # Start both servers (in real code, use threading or async)
        see_server()
        grpc_server()
        logger.info("Starting MCPServer (stub) - gRPC/SEE protocol not implemented.")
        pass

class MCPClient:
    """Multi-Agent Control Protocol Client (stub)."""
    def __init__(self, address: str):
        self.address = address
    def send(self, message: Dict[str, Any]):
        # Implement gRPC or SEE protocol client logic here
        pass

class MCPHost:
    """MCP Host for orchestrating agents (stub)."""
    def __init__(self, agents: List[Agent]):
        self.agents = agents
    def orchestrate(self):
        # Implement orchestration logic here
        pass

if __name__ == "__main__":
    ...