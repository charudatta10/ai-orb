import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid
import json
import logging

class AgentRunner:
    """
    A comprehensive runner for coordinating multiple agents to solve complex problems.
    
    The runner manages agent execution, communication, and collaborative problem-solving.
    """
    
    def __init__(
        self, 
        agents: List[Any], 
        problem_context: Dict[str, Any] = None,
        logging_level: int = logging.INFO
    ):
        """
        Initialize the multi-agent runner.
        
        Args:
            agents (List[Any]): List of agent instances to coordinate
            problem_context (Dict[str, Any], optional): Initial context for the problem
            logging_level (int): Logging configuration level
        """

        # Generate a unique run ID
        self.run_id = str(uuid.uuid4())

        # Setup logging
        self.logger = logging.getLogger(f"AgentRunner-{self.run_id}")
        self.logger.setLevel(logging_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Validate agents
        self.agents = self._validate_agents(agents)
        self.problem_context = problem_context or {}
        
        
               
        # Agent tracking
        self.agent_results = {}
        self.shared_memory = AgentSharedMemory()
    
    def _validate_agents(self, agents: List[Any]) -> List[Any]:
        """
        Validate that all agents in the list implement the required methods.
        
        Args:
            agents (List[Any]): List of agent instances
        
        Returns:
            List of validated agents
        """
        validated_agents = []
        for agent in agents:
            if hasattr(agent, 'solve_goal') and callable(agent.solve_goal):
                validated_agents.append(agent)
            else:
                self.logger.error(f"Invalid agent: {agent}. Agent must implement 'solve_goal' method.")
        return validated_agents
    
    async def run_collaborative_solve(
        self, 
        main_goal: str, 
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Collaboratively solve a problem using multiple agents.
        
        Args:
            main_goal (str): The primary objective to solve
            max_iterations (int): Maximum number of collaborative iterations
        
        Returns:
            Dict containing final problem-solving results
        """
        self.logger.info(f"Starting collaborative problem solve: {main_goal}")
        
        # Initial context preparation
        current_context = self.problem_context.copy()
        current_context['main_goal'] = main_goal
        
        for iteration in range(max_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Parallel agent execution
            agent_tasks = [
                self._run_agent(agent, current_context) 
                for agent in self.agents
            ]
            
            # Wait for all agents to complete
            agent_results = await asyncio.gather(*agent_tasks)
            
            # Aggregate and analyze results
            current_context = self._aggregate_results(
                current_context, 
                agent_results
            )
            
            # Check if goal is achieved
            if self._check_goal_completion(current_context):
                self.logger.info("Goal successfully achieved!")
                break
        
        return current_context
    
    async def _run_agent(
        self, 
        agent: Any, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single agent with the current context.
        
        Args:
            agent (Any): Agent instance to run
            context (Dict[str, Any]): Current problem context
        
        Returns:
            Agent's results and contributions
        """
        try:
            # Inject shared memory if supported
            if hasattr(agent, 'set_shared_memory'):
                agent.set_shared_memory(self.shared_memory)
            
            # Run agent's goal-solving method
            agent_result = await asyncio.to_thread(
                agent.solve_goal, 
                goal=context.get('main_goal'), 
                initial_context=context
            )
            
            self.logger.info(f"Agent {agent.__class__.__name__} completed task")
            return {
                "agent_type": agent.__class__.__name__,
                "result": agent_result
            }
        
        except Exception as e:
            self.logger.error(f"Agent {agent.__class__.__name__} failed: {e}")
            return {
                "agent_type": agent.__class__.__name__,
                "error": str(e)
            }
    
    def _aggregate_results(
        self, 
        current_context: Dict[str, Any], 
        agent_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate and merge results from multiple agents.
        
        Args:
            current_context (Dict[str, Any]): Current problem context
            agent_results (List[Dict[str, Any]]): Results from all agents
        
        Returns:
            Updated problem context
        """
        aggregated_context = current_context.copy()
        
        for result in agent_results:
            if 'result' in result:
                # Merge results into context
                agent_type = result['agent_type']
                agent_context = result['result']
                
                # Update context with agent-specific findings
                aggregated_context[f"{agent_type}_output"] = agent_context
                
                # Optionally, deep merge contexts
                if isinstance(agent_context, dict):
                    aggregated_context.update(agent_context)
        
        return aggregated_context
    
    def _check_goal_completion(
        self, 
        context: Dict[str, Any]
    ) -> bool:
        """
        Check if the main goal has been achieved.
        
        Args:
            context (Dict[str, Any]): Current problem context
        
        Returns:
            Boolean indicating goal completion
        """
        # Placeholder goal completion logic
        # Implement specific goal-checking criteria
        return all(
            context.get(f"{agent.__class__.__name__}_output") 
            for agent in self.agents
        )

class AgentSharedMemory:
    """
    Shared memory system for inter-agent communication and knowledge sharing.
    """
    def __init__(self):
        self._memory = {}
        self._lock = asyncio.Lock()
    
    async def store(self, key: str, value: Any):
        """
        Store a value in shared memory with thread-safe access.
        
        Args:
            key (str): Memory storage key
            value (Any): Value to store
        """
        async with self._lock:
            self._memory[key] = value
    
    async def retrieve(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from shared memory.
        
        Args:
            key (str): Memory storage key
            default (Any, optional): Default value if key not found
        
        Returns:
            Stored value or default
        """
        async with self._lock:
            return self._memory.get(key, default)
    
    async def clear(self):
        """
        Clear all shared memory.
        """
        async with self._lock:
            self._memory.clear()


# Allow script to be run directly or imported
if __name__ == "__main__":
    ...