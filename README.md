# LangChain — Step-by-Step Guide

Learn LangChain v1 from basics to production, using Azure OpenAI as the LLM provider.

## What's Covered

| Step | Topic |
|------|-------|
| 00 | Raw Azure OpenAI — Direct SDK call (no LangChain baseline) |
| 01 | Models — init_chat_model (universal model initializer) |
| 02 | Messages — Types, formats, and content_blocks |
| 03 | Streaming & Batch — Real-time output and parallel calls |
| 04 | Prompt Templates — Reusable dynamic prompts |
| 05 | Chains (LCEL) — Composable pipelines with pipe operator |
| 06 | Output Parsers — Structured responses (JSON, Pydantic) |
| 07 | Chain with History — Conversation memory in chains |
| 08 | Fallbacks — Graceful error recovery |
| 09 | RAG Pipeline — Retrieval-Augmented Generation |
| 10 | Tools — Function calling and tool binding |
| 11 | Agents — Autonomous reasoning with ReAct |
| 12 | Agent Streaming — Real-time agent output |
| 13 | Middleware — Custom processing layers |
| 14 | Callbacks — Hooks for logging, tracing, monitoring |
| 15 | Real World Chain — Production-ready chain patterns |

## Setup

1. Clone the repo
2. Create a virtual environment: python -m venv .venv
3. Install dependencies: pip install -r requirements.txt
4. Create a .env file with your Azure OpenAI credentials:

AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
OPENAI_API_VERSION=2025-03-01-preview


5. Run any step: python step_01_models.py

## Tech Stack

- Python 3.11+
- LangChain v1
- langchain-openai (Azure OpenAI provider)
- Azure OpenAI (GPT-4o)

langchain, azure-openai, python, llm, rag, agents, lcel, chains, ai, prompt-engineering
