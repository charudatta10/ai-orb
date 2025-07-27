# © 2025 Charudatta Korde · Licensed under CC BY-NC-SA 4.0 · View License @ https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
# C:\Users\korde\Home\Github\task-runner-SDLC\src\templates/LICENSE
"""
Multi-Agent Blog Writing System Using FastAPI and FastMACP

This system implements a multi-agent architecture for blog content creation with:
- Web scraping capabilities for research
- PDF processing for knowledge extraction
- Multiple LLM agents powered by Ollama
- FastMACP for agent communication
- FastAPI for serving the application

Structure:
1. Agent Classes and Roles
2. Communication Protocol
3. Resource Processing (Web/PDF)
4. FastAPI Implementation
5. Orchestration Logic
"""

# Required packages:
# pip install fastapi uvicorn pydantic ollama websockets beautifulsoup4 requests PyPDF2 asyncio

import os
import time
import json
import asyncio
import requests
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# For web scraping
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# For PDF processing
import PyPDF2
from io import BytesIO
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("blog_agent_system")

# Constants
OLLAMA_API_URL = "http://localhost:11434/api"
DEFAULT_MODEL = "llama3"

# ======================= 1. AGENT DEFINITIONS =======================

class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    CONTENT_WRITER = "content_writer"
    EDITOR = "editor"
    FACT_CHECKER = "fact_checker"
    ORCHESTRATOR = "orchestrator"

class MessageType(str, Enum):
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    STATUS = "status"
    ERROR = "error"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentMessage:
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: str
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    in_reply_to: Optional[str] = None

class Resource(BaseModel):
    resource_id: str
    resource_type: str  # "webpage", "pdf", "text"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BlogTask(BaseModel):
    task_id: str
    title: str
    topic: str
    keywords: List[str] = Field(default_factory=list)
    target_word_count: int = 800
    resources: List[str] = Field(default_factory=list)  # List of resource IDs
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

class Agent:
    def __init__(self, agent_id: str, role: AgentRole, model: str = DEFAULT_MODEL):
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []
        self.tasks: Dict[str, BlogTask] = {}
        self.is_busy = False

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process an incoming message and optionally return a response"""
        self.inbox.append(message)
        
        if message.message_type == MessageType.TASK:
            task_id = message.content.get("task_id")
            if task_id and task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.IN_PROGRESS
                self.tasks[task_id].updated_at = time.time()
            
            # Handle the task based on agent role
            return await self._process_task(message)
        
        elif message.message_type == MessageType.QUERY:
            # Handle queries from other agents
            return await self._process_query(message)
        
        return None

    async def _process_task(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process a task based on agent role - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")

    async def _process_query(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process a query from another agent"""
        # Default implementation - can be overridden by subclasses
        return AgentMessage(
            message_id=f"msg_{int(time.time()*1000)}",
            message_type=MessageType.RESPONSE,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            content={"error": "Query not supported by this agent"},
            in_reply_to=message.message_id
        )

    async def run_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Run inference using Ollama API"""
        if system_prompt:
            data = {
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            }
        else:
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        
        try:
            response = requests.post(f"{OLLAMA_API_URL}/generate", json=data)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error: {str(e)}"

    def add_task(self, task: BlogTask):
        """Add a task to the agent's task list"""
        self.tasks[task.task_id] = task


# =============== 2. IMPLEMENTING SPECIFIC AGENTS ===============

class ResearcherAgent(Agent):
    def __init__(self, agent_id: str, model: str = DEFAULT_MODEL):
        super().__init__(agent_id, AgentRole.RESEARCHER, model)
        
    async def _process_task(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process research tasks like web scraping and PDF analysis"""
        task_type = message.content.get("type")
        
        if task_type == "web_research":
            urls = message.content.get("urls", [])
            query = message.content.get("query", "")
            
            research_results = []
            for url in urls:
                content = await self._scrape_webpage(url)
                if content:
                    research_results.append({"url": url, "content": content})
            
            # Summarize the research if query is provided
            if query and research_results:
                summarized_results = await self._summarize_research(query, research_results)
                return AgentMessage(
                    message_id=f"msg_{int(time.time()*1000)}",
                    message_type=MessageType.RESULT,
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    content={"research_summary": summarized_results, "raw_results": research_results},
                    in_reply_to=message.message_id
                )
            
            return AgentMessage(
                message_id=f"msg_{int(time.time()*1000)}",
                message_type=MessageType.RESULT,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                content={"research_results": research_results},
                in_reply_to=message.message_id
            )
            
        elif task_type == "pdf_analysis":
            pdf_ids = message.content.get("pdf_ids", [])
            query = message.content.get("query", "")
            
            # This would need to be implemented to retrieve PDF content from a storage system
            pdf_contents = await self._analyze_pdfs(pdf_ids)
            
            return AgentMessage(
                message_id=f"msg_{int(time.time()*1000)}",
                message_type=MessageType.RESULT,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                content={"pdf_analysis": pdf_contents},
                in_reply_to=message.message_id
            )
        
        return None
    
    async def _scrape_webpage(self, url: str) -> str:
        """Scrape content from a webpage"""
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.extract()
                
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up text: remove multiple newlines and spaces
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            return text
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ""
    
    async def _analyze_pdfs(self, pdf_ids: List[str]) -> Dict[str, str]:
        """Analyze PDF documents"""
        # In a real implementation, this would retrieve PDF files from storage
        # For now, this is a placeholder
        results = {}
        return results
    
    async def _summarize_research(self, query: str, research_results: List[Dict[str, str]]) -> str:
        """Summarize research results based on a query using LLM"""
        combined_text = ""
        for result in research_results:
            # Limit the content length to avoid LLM context limits
            content = result.get("content", "")
            if len(content) > 1000:
                content = content[:1000] + "..."
            combined_text += f"Source: {result.get('url')}\n{content}\n\n"
        
        prompt = f"""Summarize the following web research results to answer this question: 
        {query}
        
        RESEARCH CONTENT:
        {combined_text}
        
        Provide a concise summary focusing on the most relevant information to answer the question.
        """
        
        summary = await self.run_llm(prompt, system_prompt="You are a research assistant synthesizing information from multiple sources.")
        return summary


class ContentWriterAgent(Agent):
    def __init__(self, agent_id: str, model: str = DEFAULT_MODEL):
        super().__init__(agent_id, AgentRole.CONTENT_WRITER, model)
        
    async def _process_task(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Write blog content based on research materials"""
        task_id = message.content.get("task_id")
        blog_title = message.content.get("title")
        topic = message.content.get("topic")
        keywords = message.content.get("keywords", [])
        target_word_count = message.content.get("target_word_count", 800)
        research_materials = message.content.get("research_materials", {})
        
        # Generate blog content using LLM
        blog_content = await self._generate_blog_content(
            title=blog_title,
            topic=topic,
            keywords=keywords,
            target_word_count=target_word_count,
            research_materials=research_materials
        )
        
        return AgentMessage(
            message_id=f"msg_{int(time.time()*1000)}",
            message_type=MessageType.RESULT,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            content={
                "task_id": task_id,
                "blog_content": blog_content,
                "title": blog_title,
                "word_count": len(blog_content.split())
            },
            in_reply_to=message.message_id
        )
    
    async def _generate_blog_content(
        self, 
        title: str, 
        topic: str, 
        keywords: List[str], 
        target_word_count: int, 
        research_materials: Dict[str, Any]
    ) -> str:
        """Generate blog content using LLM based on research materials"""
        # Prepare research information to include in the prompt
        research_summary = research_materials.get("research_summary", "")
        
        # Extract key points from raw research if available
        raw_results = research_materials.get("raw_results", [])
        raw_extracts = ""
        for result in raw_results:
            content = result.get("content", "")
            if content:
                # Take just a snippet to avoid context limit issues
                raw_extracts += f"From {result.get('url', 'source')}:\n{content[:300]}...\n\n"
        
        keywords_str = ", ".join(keywords)
        
        prompt = f"""Write a blog post with the title: "{title}"
        
        Topic: {topic}
        Target keywords: {keywords_str}
        Target word count: {target_word_count}
        
        Research summary: {research_summary}
        
        Additional source material:
        {raw_extracts}
        
        Please write a comprehensive blog post that:
        1. Has a compelling introduction
        2. Includes well-structured sections with headings
        3. Incorporates the keywords naturally
        4. Concludes with a call to action or thought-provoking question
        5. Cites information from the research materials when appropriate
        
        Format the blog with Markdown formatting for headings, bullet points, etc.
        """
        
        system_prompt = "You are a professional blog writer who creates engaging, informative content. Write in a conversational yet authoritative tone."
        
        blog_content = await self.run_llm(prompt, system_prompt=system_prompt)
        return blog_content


class EditorAgent(Agent):
    def __init__(self, agent_id: str, model: str = DEFAULT_MODEL):
        super().__init__(agent_id, AgentRole.EDITOR, model)
        
    async def _process_task(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Edit and improve blog content"""
        task_id = message.content.get("task_id")
        blog_title = message.content.get("title", "")
        blog_content = message.content.get("blog_content", "")
        keywords = message.content.get("keywords", [])
        
        # Edit the blog content
        edited_content = await self._edit_content(
            title=blog_title,
            content=blog_content,
            keywords=keywords
        )
        
        return AgentMessage(
            message_id=f"msg_{int(time.time()*1000)}",
            message_type=MessageType.RESULT,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            content={
                "task_id": task_id,
                "edited_content": edited_content,
                "title": blog_title,
                "word_count": len(edited_content.split())
            },
            in_reply_to=message.message_id
        )
    
    async def _edit_content(self, title: str, content: str, keywords: List[str]) -> str:
        """Edit and improve content using LLM"""
        keywords_str = ", ".join(keywords)
        
        prompt = f"""Edit and improve the following blog post with the title: "{title}"
        
        Target keywords: {keywords_str}
        
        ORIGINAL CONTENT:
        {content}
        
        Please make the following improvements:
        1. Fix any grammar or spelling errors
        2. Improve readability by breaking up long paragraphs and sentences
        3. Enhance the title and headings for better SEO
        4. Make sure keywords are used effectively but not overused
        5. Improve transitions between sections
        6. Make sure the introduction is engaging and the conclusion is strong
        7. Maintain the original meaning and information
        
        Return the complete edited version with all formatting intact.
        """
        
        system_prompt = "You are an expert editor who improves writing while maintaining the author's voice and the original meaning."
        
        edited_content = await self.run_llm(prompt, system_prompt=system_prompt)
        return edited_content


class FactCheckerAgent(Agent):
    def __init__(self, agent_id: str, model: str = DEFAULT_MODEL):
        super().__init__(agent_id, AgentRole.FACT_CHECKER, model)
        
    async def _process_task(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Check blog content for factual accuracy"""
        task_id = message.content.get("task_id")
        blog_content = message.content.get("blog_content", "")
        research_materials = message.content.get("research_materials", {})
        
        # Check facts in the blog content
        fact_check_results = await self._check_facts(
            content=blog_content,
            research_materials=research_materials
        )
        
        return AgentMessage(
            message_id=f"msg_{int(time.time()*1000)}",
            message_type=MessageType.RESULT,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            content={
                "task_id": task_id,
                "fact_check_results": fact_check_results,
                "all_clear": fact_check_results.get("all_clear", False)
            },
            in_reply_to=message.message_id
        )
    
    async def _check_facts(self, content: str, research_materials: Dict[str, Any]) -> Dict[str, Any]:
        """Check facts in content against research materials"""
        # Extract relevant information from research materials
        research_summary = research_materials.get("research_summary", "")
        raw_results = research_materials.get("raw_results", [])
        
        # Combine research material for context
        research_context = research_summary + "\n\n"
        for result in raw_results:
            if "content" in result:
                # Limit content to prevent exceeding context limits
                research_context += f"From {result.get('url', 'source')}:\n{result['content'][:500]}...\n\n"
        
        prompt = f"""Fact check the following blog content against the provided research materials.
        
        BLOG CONTENT:
        {content}
        
        RESEARCH MATERIALS:
        {research_context}
        
        Please identify any factual claims in the blog content that:
        1. Contradict the research materials
        2. Are not supported by the research materials
        3. Are exaggerated or misrepresented compared to the research
        
        For each issue found, provide:
        - The problematic claim
        - Why it's problematic
        - A suggested correction
        
        If everything is factually accurate, state "All facts appear to be accurate based on the provided research."
        """
        
        system_prompt = "You are a meticulous fact-checker who verifies information against reliable sources."
        
        fact_check = await self.run_llm(prompt, system_prompt=system_prompt)
        
        # Determine if the content passed the fact check
        all_clear = "All facts appear to be accurate" in fact_check
        
        return {
            "all_clear": all_clear,
            "analysis": fact_check,
            "suggestions": [] if all_clear else self._extract_suggestions(fact_check)
        }
    
    def _extract_suggestions(self, fact_check_text: str) -> List[Dict[str, str]]:
        """Extract structured suggestions from the fact check text"""
        # This is a simplified version - in a real system, you might
        # use more sophisticated parsing or ask the LLM to output in a parsable format
        suggestions = []
        
        # Simple heuristic for extracting suggestions
        sections = fact_check_text.split("- The problematic claim")
        for section in sections[1:]:  # Skip the first section (pre-amble)
            parts = section.split("- Why it's problematic", 1)
            if len(parts) > 1:
                claim = parts[0].strip()
                rest = parts[1].split("- A suggested correction", 1)
                if len(rest) > 1:
                    reason = rest[0].strip()
                    correction = rest[1].strip()
                    suggestions.append({
                        "claim": claim,
                        "reason": reason,
                        "correction": correction
                    })
        
        return suggestions


class OrchestratorAgent(Agent):
    def __init__(self, agent_id: str, model: str = DEFAULT_MODEL):
        super().__init__(agent_id, AgentRole.ORCHESTRATOR, model)
        self.agents: Dict[str, Agent] = {}
        self.resources: Dict[str, Resource] = {}
        self.blog_tasks: Dict[str, Dict[str, Any]] = {}
        
    def register_agent(self, agent: Agent):
        """Register an agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
    
    def add_resource(self, resource: Resource):
        """Add a resource to the system"""
        self.resources[resource.resource_id] = resource
    
    async def _process_task(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process orchestration tasks - management of the blog creation pipeline"""
        task_type = message.content.get("type")
        
        if task_type == "create_blog":
            # Start the blog creation process
            task_id = message.content.get("task_id", f"task_{int(time.time()*1000)}")
            blog_info = message.content.copy()
            blog_info["status"] = TaskStatus.PENDING
            blog_info["created_at"] = time.time()
            blog_info["updated_at"] = time.time()
            
            self.blog_tasks[task_id] = blog_info
            
            # Start the blog creation workflow
            asyncio.create_task(self._run_blog_creation_workflow(task_id))
            
            return AgentMessage(
                message_id=f"msg_{int(time.time()*1000)}",
                message_type=MessageType.STATUS,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                content={
                    "task_id": task_id,
                    "status": TaskStatus.IN_PROGRESS,
                    "message": "Blog creation process started"
                },
                in_reply_to=message.message_id
            )
        
        elif task_type == "check_status":
            task_id = message.content.get("task_id")
            if task_id in self.blog_tasks:
                return AgentMessage(
                    message_id=f"msg_{int(time.time()*1000)}",
                    message_type=MessageType.STATUS,
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    content={
                        "task_id": task_id,
                        "status": self.blog_tasks[task_id].get("status"),
                        "updated_at": self.blog_tasks[task_id].get("updated_at")
                    },
                    in_reply_to=message.message_id
                )
            else:
                return AgentMessage(
                    message_id=f"msg_{int(time.time()*1000)}",
                    message_type=MessageType.ERROR,
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    content={"error": f"Task {task_id} not found"},
                    in_reply_to=message.message_id
                )
        
        return None
    
    async def _run_blog_creation_workflow(self, task_id: str):
        """Run the full blog creation workflow"""
        try:
            blog_task = self.blog_tasks[task_id]
            
            # Update task status
            blog_task["status"] = TaskStatus.IN_PROGRESS
            blog_task["updated_at"] = time.time()
            
            # 1. Research phase
            researcher_id = next((agent_id for agent_id, agent in self.agents.items() 
                                if agent.role == AgentRole.RESEARCHER), None)
            if not researcher_id:
                raise ValueError("No researcher agent available")
            
            # Perform web research if URLs are provided
            urls = blog_task.get("urls", [])
            research_results = None
            if urls:
                research_message = AgentMessage(
                    message_id=f"research_task_{task_id}",
                    message_type=MessageType.TASK,
                    sender_id=self.agent_id,
                    recipient_id=researcher_id,
                    content={
                        "type": "web_research",
                        "task_id": task_id,
                        "urls": urls,
                        "query": blog_task.get("topic", "")
                    }
                )
                
                result = await self.agents[researcher_id].process_message(research_message)
                if (result and result.message_type == MessageType.RESULT):
                    research_results = result.content
            
            # Process PDFs if provided
            pdf_ids = blog_task.get("pdf_ids", [])
            pdf_results = None
            if pdf_ids:
                pdf_message = AgentMessage(
                    message_id=f"pdf_task_{task_id}",
                    message_type=MessageType.TASK,
                    sender_id=self.agent_id,
                    recipient_id=researcher_id,
                    content={
                        "type": "pdf_analysis",
                        "task_id": task_id,
                        "pdf_ids": pdf_ids,
                        "query": blog_task.get("topic", "")
                    }
                )
                
                result = await self.agents[researcher_id].process_message(pdf_message)
                if result and result.message_type == MessageType.RESULT:
                    pdf_results = result.content
            
            # Combine research materials
            research_materials = {}
            if research_results:
                research_materials.update(research_results)
            if pdf_results:
                research_materials.update(pdf_results)
            
            # 2. Content writing phase
            writer_id = next((agent_id for agent_id, agent in self.agents.items() 
                             if agent.role == AgentRole.CONTENT_WRITER), None)
            if not writer_id:
                raise ValueError("No content writer agent available")
            
            writer_message = AgentMessage(
                message_id=f"write_task_{task_id}",
                message_type=MessageType.TASK,
                sender_id=self.agent_id,
                recipient_id=writer_id,
                content={
                    "task_id": task_id,
                    "title": blog_task.get("title", ""),
                    "topic": blog_task.get("topic", ""),
                    "keywords": blog_task.get("keywords", []),
                    "target_word_count": blog_task.get("target_word_count", 800),
                    "research_materials": research_materials
                }
            )
            
            writing_result = await self.agents[writer_id].process_message(writer_message)
            if not writing_result or writing_result.message_type != MessageType.RESULT:
                raise ValueError("Failed to get content from writer agent")
            
            blog_content = writing_result.content.get("blog_content", "")
            
            # 3. Fact checking phase
            fact_checker_id = next((agent_id for agent_id, agent in self.agents.items() 
                                  if agent.role == AgentRole.FACT_CHECKER), None)
            fact_check_results = None
            if fact_checker_id:
                fact_check_message = AgentMessage(
                    message_id=f"fact_check_task_{task_id}",
                    message_type=MessageType.TASK,
                    sender_id=self.agent_id,
                    recipient_id=fact_checker_id,
                    content={
                        "task_id": task_id,
                        "blog_content": blog_content,
                        "research_materials": research_materials
                    }
                )
                
                fact_check_result = await self.agents[fact_checker_id].process_message(fact_check_message)
                if fact_check_result and fact_check_result.message_type == MessageType.RESULT:
                    fact_check_results = fact_check_result.content
            
            # 4. Editing phase
            editor_id = next((agent_id for agent_id, agent in self.agents.items() 
                            if agent.role == AgentRole.EDITOR), None)
            final_content = blog_content
            
            if editor_id:
                # If fact check failed, include the issues in the editing task
                fact_check_issues = []
                if fact_check_results and not fact_check_results.get("all_clear", False):
                    fact_check_issues = fact_check_results.get("suggestions", [])
                
                editor_message = AgentMessage(
                    message_id=f"edit_task_{task_id}",
                    message_type=MessageType.TASK,
                    sender_id=self.agent_id,
                    recipient_id=editor_id,
                    content={
                        "task_id": task_id,
                        "title": blog_task.get("title", ""),
                        "blog_content": blog_content,
                        "keywords": blog_task.get("keywords", []),
                        "fact_check_issues": fact_check_issues
                    }
                )
                
                editing_result = await self.agents[editor_id].process_message(editor_message)
                if editing_result and editing_result.message_type == MessageType.RESULT:
                    final_content = editing_result.content.get("edited_content", blog_content)
            
            # 5. Complete the task
            blog_task["status"] = TaskStatus.COMPLETED
            blog_task["updated_at"] = time.time()
            blog_task["result"] = {
                "title": blog_task.get("title", ""),
                "content": final_content,
                "word_count": len(final_content.split()),
                "fact_check": fact_check_results,
                "research_materials": {
                    "sources": [url for url in blog_task.get("urls", [])]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in blog creation workflow for task {task_id}: {str(e)}")
            blog_task["status"] = TaskStatus.FAILED
            blog_task["updated_at"] = time.time()
            blog_task["error"] = str(e)


# ======================= 3. FASTMACP IMPLEMENTATION =======================

class FastMACP:
    """Fast Multi-Agent Communication Protocol implementation"""
    
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.subscribers = {}
        self._running = False
        self._router_task = None
    
    async def start(self):
        """Start the message router"""
        if not self._running:
            self._running = True
            self._router_task = asyncio.create_task(self._message_router())
            logger.info("FastMACP message router started")
    
    async def stop(self):
        """Stop the message router"""
        if self._running:
            self._running = False
            if self._router_task:
                self._router_task.cancel()
                try:
                    await self._router_task
                except asyncio.CancelledError:
                    pass
            logger.info("FastMACP message router stopped")
    
    async def _message_router(self):
        """Route messages to their destinations"""
        while self._running:
            try:
                message = await self.message_queue.get()
                recipient_id = message.recipient_id
                
                if recipient_id in self.subscribers:
                    subscriber_queue = self.subscribers[recipient_id]
                    await subscriber_queue.put(message)
                
                self.message_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message router: {e}")
    
    async def send_message(self, message: AgentMessage):
        """Send a message to the message queue"""
        await self.message_queue.put(message)
    
    async def subscribe(self, agent_id: str) -> asyncio.Queue:
        """Subscribe an agent to receive messages"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = asyncio.Queue()
        return self.subscribers[agent_id]
    
    def unsubscribe(self, agent_id: str):
        """Unsubscribe an agent from receiving messages"""
        if agent_id in self.subscribers:
            del self.subscribers[agent_id]


# ======================= 4. FASTAPI IMPLEMENTATION =======================

app = FastAPI(title="Multi-Agent Blog Writing System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the MACP system
macp = FastMACP()

# Initialize agents
researcher = ResearcherAgent("researcher_1")
writer = ContentWriterAgent("writer_1")
editor = EditorAgent("editor_1")
fact_checker = FactCheckerAgent("fact_checker_1")
orchestrator = OrchestratorAgent("orchestrator_1")

# Register agents with orchestrator
orchestrator.register_agent(researcher)
orchestrator.register_agent(writer)
orchestrator.register_agent(editor)
orchestrator.register_agent(fact_checker)

# API Models
class BlogRequestModel(BaseModel):
    title: str
    topic: str
    keywords: List[str] = Field(default_factory=list)
    target_word_count: int = 800
    urls: List[str] = Field(default_factory=list)
    pdf_ids: List[str] = Field(default_factory=list)

class TaskStatusModel(BaseModel):
    task_id: str

class BlogResponseModel(BaseModel):
    task_id: str
    status: str
    message: str

class ResourceUploadModel(BaseModel):
    resource_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WebpageResourceModel(BaseModel):
    url: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    await macp.start()
    logger.info("Blog writing system started")

    # Start agent task processing as background tasks
    agent_tasks = [
        asyncio.create_task(run_agent_task_processing(researcher)),
        asyncio.create_task(run_agent_task_processing(writer)),
        asyncio.create_task(run_agent_task_processing(editor)),
        asyncio.create_task(run_agent_task_processing(fact_checker)),
        asyncio.create_task(run_orchestrator_task_processing(orchestrator)),
    ]
    try:
        yield
    finally:
        # Cancel all agent tasks on shutdown
        for task in agent_tasks:
            task.cancel()
        await macp.stop()
        logger.info("Blog writing system stopped")

app.router.lifespan_context = lifespan

# API Endpoints
@app.post("/blogs", response_model=BlogResponseModel)
async def create_blog(blog_request: BlogRequestModel):
    """Create a new blog post"""
    task_id = f"blog_{int(time.time()*1000)}"
    
    # Create a task message for the orchestrator
    message = AgentMessage(
        message_id=f"api_task_{task_id}",
        message_type=MessageType.TASK,
        sender_id="api",
        recipient_id="orchestrator_1",
        content={
            "type": "create_blog",
            "task_id": task_id,
            "title": blog_request.title,
            "topic": blog_request.topic,
            "keywords": blog_request.keywords,
            "target_word_count": blog_request.target_word_count,
            "urls": blog_request.urls,
            "pdf_ids": blog_request.pdf_ids
        }
    )
    
    # Send the message to the orchestrator
    await macp.send_message(message)
    
    return BlogResponseModel(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Blog creation task submitted"
    )

@app.get("/blogs/{task_id}", response_model=Dict[str, Any])
async def get_blog_status(task_id: str):
    """Get the status of a blog creation task"""
    # Create a status check message for the orchestrator
    message = AgentMessage(
        message_id=f"status_check_{int(time.time()*1000)}",
        message_type=MessageType.TASK,
        sender_id="api",
        recipient_id="orchestrator_1",
        content={
            "type": "check_status",
            "task_id": task_id
        }
    )
    
    # Create a queue to receive the response
    response_queue = asyncio.Queue()
    
    # Create a unique ID for this request
    request_id = f"request_{int(time.time()*1000)}"
    
    # Subscribe to receive the response
    macp.subscribers[request_id] = response_queue
    
    # Modify the message to send the response to our temporary subscriber
    message.recipient_id = "orchestrator_1"
    message.sender_id = request_id
    
    # Send the message
    await macp.send_message(message)
    
    try:
        # Wait for the response with a timeout
        response = await asyncio.wait_for(response_queue.get(), timeout=10.0)
        
        if response.message_type == MessageType.ERROR:
            raise HTTPException(status_code=404, detail=response.content.get("error", "Task not found"))
        
        # Check if the task is completed
        if response.message_type == MessageType.STATUS:
            status = response.content.get("status")
            
            # If completed, get the full task data from the orchestrator
            if status == TaskStatus.COMPLETED:
                return orchestrator.blog_tasks.get(task_id, {"status": "Unknown"})
            
            # Otherwise, just return the status information
            return response.content
        
        return {"error": "Unexpected response"}
    
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timed out waiting for response")
    finally:
        # Unsubscribe our temporary subscriber
        macp.unsubscribe(request_id)

@app.post("/resources/webpage")
async def add_webpage_resource(resource: WebpageResourceModel):
    """Add a webpage resource for research"""
    resource_id = f"webpage_{int(time.time()*1000)}"
    
    # Create a background task to scrape and add the resource
    async def scrape_and_add():
        try:
            # Use the researcher agent to scrape the webpage
            content = await researcher._scrape_webpage(resource.url)
            
            # Create and add the resource
            new_resource = Resource(
                resource_id=resource_id,
                resource_type="webpage",
                content=content,
                metadata={
                    "url": resource.url,
                    **resource.metadata
                }
            )
            
            orchestrator.add_resource(new_resource)
            logger.info(f"Added webpage resource {resource_id}")
            
        except Exception as e:
            logger.error(f"Error adding webpage resource: {e}")
    
    # Start the background task
    background_tasks = BackgroundTasks()
    background_tasks.add_task(scrape_and_add)
    
    return {"resource_id": resource_id, "status": "processing"}

@app.post("/resources/pdf")
async def add_pdf_resource(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = ""
):
    """Upload a PDF file as a resource"""
    resource_id = f"pdf_{int(time.time()*1000)}"
    
    # Parse metadata if provided
    meta_dict = {}
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError:
            meta_dict = {"description": metadata}
    
    # Create a background task to process and add the PDF
    async def process_and_add_pdf():
        try:
            # Read the PDF content
            content = await file.read()
            
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(content)
            
            # Create and add the resource
            new_resource = Resource(
                resource_id=resource_id,
                resource_type="pdf",
                content=pdf_text,
                metadata={
                    "filename": file.filename,
                    **meta_dict
                }
            )
            
            orchestrator.add_resource(new_resource)
            logger.info(f"Added PDF resource {resource_id}")
            
        except Exception as e:
            logger.error(f"Error adding PDF resource: {e}")
    
    # Start the background task
    background_tasks.add_task(process_and_add_pdf)
    
    return {"resource_id": resource_id, "status": "processing"}

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from a PDF file"""
    try:
        # Create a PDF reader object
        pdf_file = BytesIO(pdf_content)
        reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from each page
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n\n"
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

@app.websocket("/ws/agent/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for direct agent communication"""
    await websocket.accept()
    
    # Subscribe to receive messages for this agent
    queue = await macp.subscribe(f"ws_{agent_id}")
    
    # Handle incoming messages from WebSocket and outgoing messages to WebSocket
    try:
        # Create background task to listen for messages from the queue
        async def listen_for_messages():
            while True:
                message = await queue.get()
                await websocket.send_json(message.__dict__)
        
        listener_task = asyncio.create_task(listen_for_messages())
        
        # Listen for messages from the WebSocket
        while True:
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
                
                # Create an agent message
                message = AgentMessage(
                    message_id=message_data.get("message_id", f"ws_msg_{int(time.time()*1000)}"),
                    message_type=message_data.get("message_type", MessageType.TASK),
                    sender_id=f"ws_{agent_id}",
                    recipient_id=message_data.get("recipient_id", "orchestrator_1"),
                    content=message_data.get("content", {})
                )
                
                # Send the message
                await macp.send_message(message)
                
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
            except Exception as e:
                await websocket.send_json({"error": str(e)})
    
    except WebSocketDisconnect:
        # Clean up
        if listener_task and not listener_task.done():
            listener_task.cancel()
        macp.unsubscribe(f"ws_{agent_id}")
    

# ======================= 5. MAIN APPLICATION ENTRY POINT =======================

# Agent task processing loop
async def run_agent_task_processing(agent: Agent):
    """Run an agent's task processing loop"""
    if isinstance(agent, OrchestratorAgent):
        # The orchestrator is handled differently
        return
    
    # Subscribe the agent to receive messages
    queue = await macp.subscribe(agent.agent_id)
    
    while True:
        try:
            # Wait for a message
            message = await queue.get()
            
            # Process the message
            response = await agent.process_message(message)
            
            # Send the response if there is one
            if response:
                await macp.send_message(response)
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in agent {agent.agent_id} task processing: {e}")

# Orchestrator task processing loop (modified to work with the MACP)
async def run_orchestrator_task_processing(orchestrator: OrchestratorAgent):
    """Run the orchestrator's task processing loop"""
    # Subscribe the orchestrator to receive messages
    queue = await macp.subscribe(orchestrator.agent_id)
    
    while True:
        try:
            # Wait for a message
            message = await queue.get()
            
            # Process the message
            response = await orchestrator.process_message(message)
            
            # Send the response if there is one
            if response:
                await macp.send_message(response)
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in orchestrator task processing: {e}")

# ======================= MCP SERVER WEB SCRAPER AGENT (AGENT2AGENT PROTOCOL) =======================

# Install: pip install fastmcp jrpc-sse

from fastmcp import MCPServer
from jrpc_sse import JRPCSSEServer  # For SSE/JRPC support
from typing import Any

class WebScraperServer(MCPServer):
    def scrape(self, url: str) -> str:
        """Scrapes the given URL and returns text content."""
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove script/style/meta/link for cleaner text
        for tag in soup(['script', 'style', 'meta', 'link']):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

# Agent Card for agent2agent protocol
AGENT_CARD = {
    "id": "webscraper-agent-1",
    "name": "WebScraperAgent",
    "description": "Scrapes web pages and returns clean text content.",
    "protocols": ["mcp", "jrpc", "sse"],
    "tools": [
        {
            "name": "scrape",
            "description": "Scrape a web page and return its text content.",
            "input_schema": {"url": "string"},
            "output_schema": {"content": "string"}
        }
    ]
}

def run_webscraper_server():
    # Optionally, use JRPCSSEServer for SSE/JRPC support
    # server = JRPCSSEServer()
    server = WebScraperServer(agent_card=AGENT_CARD)
    server.register_tool("scrape", server.scrape)
    # Optionally, expose agent card at /agent-card endpoint
    @server.route("/agent-card")
    def agent_card_endpoint() -> Any:
        return AGENT_CARD
    server.run(host="0.0.0.0", port=5000)

# If you want to run the MCP server as a separate process, you can add:
if __name__ == "__main__":
    # ...existing FastAPI/agent system startup...
    # To run the MCP server, uncomment the following line:
    # run_webscraper_server()
    pass