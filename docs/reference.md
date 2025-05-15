# AI-Agent Framework Technical Reference
==========================

Table of Contents
-----------------

1. [API Documentation](#api-documentation)
2. [Configuration Options](#configuration-options)
3. [Command-Line Interface Reference](#command-line-interface-reference)
4. [File Formats and Data Structures](#file-formats-and-data-structures)
5. [Architectural Overview](#architectural-overview)

## API Documentation
-------------------

The AI-Agent framework provides a RESTful API for interacting with the agent, tools, and sandbox features. The following endpoints are currently supported:

### GET / agents

* Returns a list of all available agent classes.
* Response: JSON object containing a list of agent classes.

Example:
```bash
curl -X GET 'http://localhost:5000/agents'
```

### POST / agents/{agentClass}

* Creates a new instance of the specified agent class.
* Request Body: JSON object containing configuration options for the agent.
* Response: JSON object containing the created agent ID and configuration.

Example:
```bash
curl -X POST 'http://localhost:5000/agents/LLMAgent' \
  -H 'Content-Type: application/json' \
  -d '{"config": {"learning_rate": 0.01, "batch_size": 32}}'
```

### GET / tools

* Returns a list of all available tool classes.
* Response: JSON object containing a list of tool classes.

Example:
```bash
curl -X GET 'http://localhost:5000/tools'
```

### POST / tools/{toolClass}

* Creates a new instance of the specified tool class.
* Request Body: JSON object containing configuration options for the tool.
* Response: JSON object containing the created tool ID and configuration.

Example:
```bash
curl -X POST 'http://localhost:5000/tools/MLTool' \
  -H 'Content-Type: application/json' \
  -d '{"config": {"model_path": "/path/to/model.json"}}'
```

## Configuration Options
------------------------

The AI-Agent framework allows for customization through configuration options. The following configurations are currently supported:

### Agent Configurations

* `learning_rate`: float, default=0.01, describes the learning rate of the agent.
* `batch_size`: int, default=32, describes the batch size used by the agent.

Example:
```bash
{
  "config": {
    "learning_rate": 0.05,
    "batch_size": 64
  }
}
```

### Tool Configurations

* `model_path`: string, required, describes the path to the tool's model file.
* `max_queries`: int, default=100, describes the maximum number of queries allowed for the tool.

Example:
```bash
{
  "config": {
    "model_path": "/path/to/model.json",
    "max_queries": 200
  }
}
```

## Command-Line Interface Reference
-----------------------------------

The AI-Agent framework provides a command-line interface for interacting with the agent, tools, and sandbox features. The following commands are currently supported:

### invoke

* Runs the project and executes the specified command.

Example:
```bash
invoke --agent LLMAgent --tool MLTool --query "hello world"
```

### config

* Displays the current configuration options for the agent or tool.
* Request Body: JSON object containing configuration options (optional).

Example:
```bash
curl -X GET 'http://localhost:5000/config' \
  -H 'Content-Type: application/json'
```

## File Formats and Data Structures
----------------------------------

The AI-Agent framework supports the following file formats:

### JSON

* Used for configuration options, agent data, and tool data.
* Response format: JSON object.

Example:
```json
{
  "config": {
    "learning_rate": 0.05,
    "batch_size": 64
  }
}
```

### TOML

* Used for project configuration files (pyproject.toml).
* Response format: TOML string.

Example:
```toml
[project]
name = "AI-Agent"
version = "1.0"

[tool]
name = "python"
version = "3.9.7"
```

## Architectural Overview
-------------------------

The AI-Agent framework is designed as a microservices architecture, with the following components:

* **Agent**: Manages the interaction between the user and the tools.
	+ Responsible for executing queries and receiving responses from the tools.
* **Tools**: Provide specialized functionality for tasks such as text classification, sentiment analysis, and more.
	+ Designed to be modular and extensible, allowing new tool classes to be added easily.
* **Sandbox**: Provides a testing environment for the tools and agent.
	+ Allows developers to test and iterate on their tool implementations.

The AI-Agent framework uses the following technologies:

* **Python**: Used as the primary programming language for developing the agent and tools.
* **Docker**: Used for containerization and deployment of the project.
* **Flask**: Used as a lightweight web framework for serving the API endpoints.