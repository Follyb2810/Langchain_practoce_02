from langchain.tools import tool
from langchain.agents import load_agent
from langchain_ollama import ChatOllama


# Define tools
@tool
def add(x: float, y: float) -> float:
    """Add two numbers"""
    return x + y


@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers"""
    return x * y


# LLM
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# ❌ Old API: load a prebuilt agent with default prompt
# "zero-shot-react-description" = classic ReAct style
agent = load_agent("zero-shot-react-description", llm=llm, tools=[add, multiply])

# Run
result = agent.run("What is 5 multiplied by 12, then add 3?")
print(result)
