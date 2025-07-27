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



@dataclass
class Observe:
    """Observe module manages memory and observations."""
    # Using field(default_factory=list) to avoid mutable default argument problems
    observations: List[Dict[str, Any]] = field(default_factory=list)
    past_plans: List[Dict[str, Any]] = field(default_factory=list)

    def record_observation(self, observation: Dict[str, Any]) -> None:
        """
        Record an observation from an action.

        Args:
            observation: Observation details
        """
        observation_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            **observation
        }
        self.observations.append(observation_with_timestamp)

    def record_plan(self, plan: List[Dict[str, Any]]) -> None:
        """
        Record a plan for future reference.

        Args:
            plan: Plan steps
        """
        self.past_plans.append({
            "timestamp": datetime.now().isoformat(),
            "plan": plan
        })
        
    def get_recent_observations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent observations.
        
        Args:
            limit: Maximum number of observations to return
            
        Returns:
            Recent observations
        """
        return self.observations[-limit:] if self.observations else []