# Contribution Guide for AI Agent Project

Welcome to the AI Agent project contribution guide! We're glad you're interested in helping us improve and expand this exciting project. Below, you'll find a step-by-step guide on how to report issues, create pull requests, follow coding standards, test requirements, and more.

## Reporting Issues
If you've found a bug or have a feature request, please don't hesitate to open an issue on our GitHub page: [https://github.com/charudatta10/ai-agent/issues](https://github.com/charudatta10/ai-agent/issues). When reporting an issue, please provide as much detail as possible, including:

* A clear description of the problem or request
* Any relevant error messages or logs
* Steps to reproduce the issue (if applicable)

## Pull Request Process
When creating a pull request, ensure you follow these guidelines:

1. **Create a new branch**: Fork our repository and create a new branch from `main` using the following command:
```bash
git checkout -b feature/new-feature
```
2. **Make changes**: Make your desired changes to the codebase.
3. **Write tests**: Write unit tests or integration tests for your changes, if applicable.
4. **Run tests**: Run all tests in the `tests` directory using:
```bash
pytest
```
5. **Commit changes**: Commit your changes with a clear and descriptive commit message.

Example commit message:
```
feat: Add new feature to AI agent

This commit adds a new feature to the AI agent, including [new code] and [test files].
```

6. **Create a pull request**: Once you've committed all changes, create a pull request against `main`.

## Coding Standards
To ensure consistency throughout our project, we follow these coding standards:

*   Follow PEP 8 for Python coding style.
*   Use Black to format code (install with `pip install black`).
*   Keep imports concise and organized.

Example code snippet:
```python
# Good practice

import os
from typing import Dict

def process_data(data: Dict) -> None:
    # Process data here
```

Example bad practice:
```python
# Avoid this

import os, sys, threading

def process_data() -> None:
    # Process data here
```

## Testing Requirements
Our project relies on unit tests and integration tests. To ensure your changes are thoroughly tested:

*   Write unit tests using Pytest.
*   Use a testing framework like `pytest` or `unittest`.
*   Run all tests in the `tests` directory using:
```bash
pytest
```

## Documentation Guidelines
To maintain our project's documentation, please follow these guidelines:

*   Use Markdown format for all documents.
*   Keep all text concise and accurate.
*   Follow our README.md template as an example.

Example README snippet:
```markdown
# AI Agent Project

A high-performance AI agent framework built using Python and Ollama.
```

## Additional Tips
Before submitting your pull request, make sure to:

*   Check the project's LICENSE file for any restrictions on modifications or usage.
*   Read our README.md file for additional guidelines and information.

By following this contribution guide, you'll help us create a better AI Agent project for everyone. If you have any questions or concerns, feel free to open an issue or reach out to us via GitHub discussions!