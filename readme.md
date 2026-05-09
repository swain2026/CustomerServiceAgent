# Customer Service Agent

A Python-based intelligent customer service agent built with LangGraph and DeepSeek.

## Overview

This project implements an intelligent customer service system capable of understanding user intent, analyzing emotions, and engaging in multi-turn conversations.

## Core Components

- **LangGraph**: Enables building complex, retryable, and interruptible agent workflows
- **DeepSeek**: Provides powerful language understanding and generation capabilities
- **Intelligent Customer Service**: Features intent recognition, sentiment analysis, and multi-turn dialogue management

## Features

- 🤖 Natural language understanding and response generation
- 📊 Sentiment analysis for customer emotion detection
- 🔄 Multi-turn conversation context management
- 🔄 Retryable and interruptible workflow execution

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Copy `.env.example` to `.env` and configure your DeepSeek API key:
```bash
cp .env.example .env
```

5. Run the application:
```bash
customer-service-agent
```

## Project Structure

```
CustomerServiceAgent/
├── customer_service_agent/
│   ├── __init__.py
│   ├── main.py
│   ├── agent.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   └── ...
│   └── edges/
│       ├── __init__.py
│       └── ...
├── tests/
│   ├── __init__.py
│   └── ...
├── pyproject.toml
├── .env.example
└── readme.md
```

## Development

Run tests:
```bash
pytest
```

Run linter:
```bash
ruff check .
```

## License

MIT