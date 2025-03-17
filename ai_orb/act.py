import json
from typing import Any, Dict, List, Callable, Optional, Union
from dataclasses import dataclass, field
from ai_orb.sandbox import SecureSandbox, Tool
import ollama
import logging
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class Act:
    """Act module handles the execution of tasks in a secure sandbox."""
    sandbox: SecureSandbox

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
            result = self.sandbox.execute(tool_name, input_data)
            return {"success": True, "output": result}
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return {"success": False, "error": str(e)}