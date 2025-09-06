from typing import Dict, Any, Optional
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool


# === 1. Define tools ===
@tool
def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y

@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y

@tool
def exponentiate(x: float, y: float, z: Optional[float] = None) -> float:
    """Raise x to the power of y."""
    return x ** y

@tool
def subtract(x: float, y: float) -> float:
    """Subtract x from y."""
    return y - x


# === 2. Define LLM ===
llm: ChatOllama = ChatOllama(
    model="mistral", 
    base_url="http://localhost:11434"
)


# === 3. Memory ===
# Stores conversation history, so the agent "remembers".
memory: ConversationBufferMemory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)


# === 4. Prompt template ===
chat_template: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}") 
])


# === 5. Create agent and executor ===
tools = [add, subtract, multiply, exponentiate]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=chat_template
)

agent_executor: AgentExecutor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)


# === 6. Run agent ===
query: Dict[str, Any] = {"input": "what is 10.07 multiplied by 7.687"}
response: Dict[str, Any] = agent_executor.invoke(query)

print("Agent Response:", response)

