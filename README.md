# AI Data Analysis Agent Collection

This repository contains four AI agent projects for CSV and tabular data analysis. The projects range from a minimal CSV question-answering agent to LangGraph-based planning, interactive visualization, conversation memory, validation, and multi-dataset orchestration.


## Projects

### 1. CSV_QA_Agent

A minimal CSV question-answering agent. It reads the CSV schema, asks an OpenAI model to generate pandas analysis code for a user question, executes the generated code, evaluates whether the result is usable, and formats the final answer.

Key features:

- Inspects CSV columns, data types, and sample rows
- Generates pandas analysis code with OpenAI
- Executes generated code and reads the `result` variable
- Detects execution errors and empty results
- Includes sample and interactive test modes


### 2. Plan-and-Execute_Data_Analysis_Agent

A LangGraph-based plan-and-execute data analysis agent. The LLM produces a structured JSON plan, while deterministic pandas operators execute the actual data operations.

Key features:

- LangGraph workflow: `plan -> execute -> check -> retry -> answer`
- Structured JSON planning instead of direct natural-language execution
- Deterministic operators for filtering, grouping, aggregation, sorting, ranking, column derivation, and row limits
- Execution traces showing input and output row/column counts for each step
- Built-in test runner with 15 basic, intermediate, and advanced questions


### 3. Interactive__Agent_with_Visualization_and_Memory

An interactive CSV QA agent with a FastAPI backend, a browser frontend, conversation memory, and Plotly visualization support.

Key features:

- FastAPI endpoints: `/chat`, `/history`, `/reset`, `/datasets`, `/health`
- Server-Sent Events for progress updates and final results
- Conversation memory for follow-up questions
- Automatic visualization decision and Plotly chart generation
- Includes `housing.csv` and `Netflix.csv` datasets


### 4. Orchestrated_AI_Analytics_System

A more complete orchestrated analytics system. It first classifies the user question as either a generic knowledge question or a data analysis question. Generic questions are routed to DuckDuckGo search, while analytics questions are routed to an iterative analytics agent and independently checked by a validator agent.

Key features:

- LangGraph orchestrator for `generic` and `analytics` routing
- DuckDuckGo search for general knowledge questions
- Iterative analytics agent with a persistent execution namespace
- Independent validator agent that generates and runs its own validation plan
- Retry loop when validation returns incomplete or suspicious results
- Multi-year financial datasets
- Plotly chart generation for frontend rendering

Run the backend:

```bash
cd Orchestrated_AI_Analytics_System
uvicorn backend.main:app --reload --port 8000
```

## Requirements

Python 3.10 or later is recommended.

Install the full dependency set:

```bash
pip install openai pandas numpy langgraph typing-extensions fastapi uvicorn pydantic plotly python-dotenv langchain-community langchain-openai duckduckgo-search
```

For the minimal `CSV_QA_Agent`, the core dependencies are:

```bash
pip install openai pandas
```

## Environment Variables

Each project reads OpenAI configuration from a `.env` file in its own project directory.

Example `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=gpt-4.1-mini
