# Customer Service Agent

A Python-based intelligent customer service agent built with LangGraph and OpenAI-compatible models (including DeepSeek).

## Overview

This project implements an intelligent customer service system capable of understanding user intent, analyzing emotions, and engaging in multi-turn conversations with optional human agent escalation.

## Core Components

- **LangGraph**: Enables building complex, retryable, and interruptible agent workflows
- **OpenAI-compatible API**: Supports DeepSeek and other OpenAI-compatible models
- **Pinecone**: Vector database for knowledge base retrieval
- **Intelligent Customer Service**: Features intent recognition, sentiment analysis, and multi-turn dialogue management

## Features

- 🤖 Natural language understanding and response generation
- 📊 Sentiment analysis for customer emotion detection (积极/中性/消极/愤怒)
- 🎯 Intent classification (咨询类/投诉类/售后类/订单类/技术支持/其他)
- 🔄 Multi-turn conversation context management
- 🔄 Retryable and interruptible workflow execution
- 👥 Human agent escalation for complex issues

## Architecture

The agent workflow consists of the following nodes:

1. **classify_intent**: Classifies user input into one of six intent categories
2. **analyze_emotion**: Analyzes user emotion state
3. **generate_response**: Generates initial response using the language model
4. **tools**: Executes tool calls (user info, orders, product info, knowledge base)
5. **human_agent**: Escalates to human agent when needed

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

4. Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```

5. Run the application:
```bash
python main.py
```

## Project Structure

```
CustomerServiceAgent/
├── main.py              # Entry point for the agent
├── agent.py             # CustomerServiceAgent class for session management
├── agentstate.py        # AgentState TypedDict definition
├── nodes.py             # Workflow nodes (intent, emotion, response, tools)
├── edges.py             # Workflow edge conditions
├── workflow.py          # Graph workflow definition
├── model.py             # Model configuration (thinking/classification models)
├── tools.py             # Tool definitions (user info, orders, products, knowledge base)
├── pinecone.py          # Pinecone vector database integration
├── pyproject.toml       # Project configuration
├── .env.example         # Environment variables template
└── readme.md
```

## Environment Variables

Required variables in `.env`:

```env
# Model Configuration
MODEL_BASE_URL=your_api_base_url
MODEL_API_KEY=your_api_key
MODEL_NAME=your_model_name
EM_MODEL_NAME=embedding_model_name

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=customer-service-kb
```

## Supported Tools

- **get_user_info**: Retrieves user information by user_id
- **get_latest_order**: Gets the most recent order for a user
- **get_product_info**: Retrieves product details and pricing
- **retrieve_knowledge_base**: Searches Pinecone for relevant knowledge base information

## Intent Categories

- 咨询类 (Inquiry)
- 投诉类 (Complaint)
- 售后类 (After-sales)
- 订单类 (Order)
- 技术支持 (Technical Support)
- 其他 (Other)

## Emotion Categories

- 积极 (Positive)
- 中性 (Neutral)
- 消极 (Negative)
- 愤怒 (Angry)

## Human Escalation

The agent automatically escalates to human agents when:
- User emotion is 愤怒 or 消极 with 投诉类 or 技术支持 intent
- User input exceeds 500 characters (complex issue)
- Other complex scenarios requiring human intervention


## test examples

- "我想了解一下你们的产品价格",
- "这个智能手表多少钱，有什么优惠活动呢",
- "我买的商品有问题，要求退货",
- "这个产品怎么这么差劲，我要投诉！",
- "我的订单什么时候能发货？"


## Development

Run linter:
```bash
ruff check .
```

## License

MIT