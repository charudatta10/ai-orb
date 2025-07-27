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


class Act:
    """Act module handles the execution of tasks in a secure sandbox."""
    def __init__(self, sandbox: SecureSandbox):
        self.sandbox = sandbox

    def execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool securely in the sandbox.

        Args:
            tool_name: The name of the tool to execute
            input_data: Input for the tool

        Returns:
            Result of the tool execution
        """
        try:
            # Look up the tool in the sandbox's allowed tools
            if hasattr(self.sandbox, "tools") and tool_name in self.sandbox.tools:
                tool = self.sandbox.tools[tool_name]
                result = tool(input_data)
            else:
                # Fallback: try to execute by name (for protocol compatibility)
                result = self.sandbox.execute(tool_name, input_data)
            return {"success": True, "output": result}
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return {"success": False, "error": str(e)}