# Getting Started with AI Agent Framework
======================================

## Prerequisites and System Requirements
--------------------------------------

### Operating System

*   Windows 10 (64-bit) or later
*   macOS High Sierra (or later)
*   Linux distributions supported by Python 3.8 or later

### Python Version

*   Python 3.8 or later (required for the framework's functionality)

### Additional Requirements

*   Git version 2.24 or later
*   A code editor or IDE of your choice
*   Basic knowledge of Python programming language

## Installation Instructions
-------------------------

### Clone the Repository

1.  Open a terminal or command prompt.
2.  Navigate to the desired directory where you want to clone the repository.
3.  Run the following command to clone the AI Agent Framework repository:

    ```bash
git clone https://github.com/charudatta10/ai-agent.git
```

### Install Dependencies

1.  Navigate into the cloned repository:
    ```bash
cd ai-agent
```
2.  Create a virtual environment using Python's `venv` module to isolate dependencies for the project:

    ```bash
python -m venv env
```
3.  Activate the virtual environment (for Windows):
    ```cmd
env\Scripts\activate
```

    (for macOS/Linux):
    ```bash
source env/bin/activate
```
4.  Install the required dependencies using pip:
    ```bash
pip install -r requirements.txt
```

## Basic Configuration
---------------------

### Environment Variables

1.  Create a `.env` file in the project root directory to store environment-specific variables:

    ```bash
touch .env
```

    Edit the `.env` file to add your environment variables, for example:

    ```
AI_AGENT_API_KEY=your-api-key
AI_AGENT_SECRET_KEY=your-secret-key
```
2.  Add these variables to your system's environment or use a tool like `dotenv` to load them automatically.

### Configuration File

1.  The AI Agent Framework uses the configuration file located at `~/.config/ai-agent/config.yaml`.

## Running a Simple Example
---------------------------

1.  Create an instance of the AI agent using Python:

    ```python
from ai_agent import AIAgent

# Initialize with configuration file
agent = AIAgent(config_path='~/.config/ai-agent/config.yaml')

# Run the agent
agent.run()
```

2.  Alternatively, you can invoke the project directly without initializing an instance of the AI Agent:

```bash
invoke
```

## Where to Go Next
------------------

*   **Explore Features**: Browse through the project's features and explore how they contribute to the framework's functionality.
*   **Contribute to the Project**: Submit issues, feature requests, or pull requests for new features or bug fixes.
*   **Learn More About AI Agent Framework**: Visit the [Documentation](https://github.com/charudatta10/ai-agent/blob/main/README.md) and explore other resources provided by the project's creators.

By following these steps, you can successfully set up and start exploring the AI Agent Framework.