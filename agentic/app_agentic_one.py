from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.base import RunnableSerializable

# ==============================
# DEFINE TOOLS
# ==============================
@tool
def add(x: float, y: float) -> float:
    """Add two numbers"""
    return x + y

@tool
def subtract(x: float, y: float) -> float:
    """Subtract y from x"""
    return x - y

@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers"""
    return x * y

@tool
def divide(x: float, y: float) -> float:
    """Divide x by y"""
    return x / y

@tool
def final_answer(answer: str, tools_used: list[str]) -> str:
    """Use this tool to provide a final answer to the user.
    The answer should be in natural language as this will be provided
    to the user directly. The tools_used must include a list of tool
    names that were used within the `scratchpad`.
    """
    return {"answer": answer, "tools_used": tools_used}


# ==============================
# DEFINE LLM
# ==============================
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# ==============================
# DEFINE PROMPT
# ==============================
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You're a helpful assistant. When answering a user's question "
                "you should first use one of the tools provided. After using a "
                "tool the tool output will be provided in the "
                "'scratchpad' below. If you have the answer in the "
                "scratchpad you should not use any tools and "
                "instead answer directly to the user."
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# ==============================================================
# WAY 1: High-level Agent (create_tool_calling_agent + AgentExecutor)
# ==============================================================

tools = [add, subtract, multiply, divide]

# Create the agent definition
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

# Wrap the agent in an executor (handles reasoning loop + memory)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent (LangChain manages scratchpad, calling tools, returning answer)
print("=== AgentExecutor run ===")
response = executor.invoke({"input": "what is 10 + 10?", "chat_history": []})
print(response)


# ==============================================================
# WAY 2: Low-level LCEL Runnable graph
# ==============================================================

# Here, YOU wire prompt → llm → tool binding manually
agentic: RunnableSerializable = (
    {
        "input": lambda x: x["input"],            # extract user input
        "chat_history": lambda x: x["chat_history"],  # pass history
        "agent_scratchpad": lambda x: x.get("agent_scratchpad", []), # pass scratchpad
    }
    | prompt                                     # format into a chat prompt
    | llm.bind_tools(tools, tool_choice="any")   # bind tools manually
)

# Run the chain (but YOU handle inputs/outputs)
print("=== Runnable graph run ===")
tool_call = agentic.invoke({"input": "what is 10 + 10", "chat_history": []})
print(tool_call)
