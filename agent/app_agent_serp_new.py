from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

# Define tools (same as before)
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

# Memory (conversation state across turns)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt template with placeholders
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful math assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# ✅ New API: build tool-calling agent with YOUR prompt
agent = create_tool_calling_agent(llm=llm, tools=[add, multiply], prompt=chat_template)

# Wrap it with AgentExecutor so memory works automatically
agent_executor = AgentExecutor(agent=agent, tools=[add, multiply], memory=memory, verbose=True)

# Run
response = agent_executor.invoke({"input": "What is 5 multiplied by 12, then add 3?"})
print(response)
