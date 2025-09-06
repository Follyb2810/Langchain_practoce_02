###### https://www.aurelio.ai/learn/langchain-conversational-memory

### pip install jupyterlab langchain langgraph fastapi uvicorn transformers torch

### run bash jupyter lab

### notebooks Jupyter experiments (LangGraph workflows, tests)

### app FastAPI backend (API for React frontend)

### data Triage guideline docs

### requirements.txt

### Dockerfile

### docker run -d --name ollama -p 11434:11434 ollama/ollama

### docker exec -it ollama ollama pull mistral

```bash
React (frontend)
   ⬇️ user interacts
FastAPI (Python backend)
   ⬇️ LangGraph Orchestration
LangGraph nodes:
   - greet_node
   - fetch_patient_node (query DB: MongoDB/Postgres)
   - followup_node (ask clarifying questions)
   - vitals_node (request vitals if needed)
   - triage_node (LLM classification using guideline embeddings)
Ollama (local LLM: mistral/gemma/llama3)
Vector DB (Mongo Atlas, Chroma, or Qdrant for guidelines)
SQL/NoSQL DB (patient history)

```
# Langchain_practoce_02
