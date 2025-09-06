# Import necessary modules from LangChain and Ollama integration
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.messages import ToolMessage

# ==========================
# 1. Define Tools
# ==========================


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
    """
    Tool to provide the FINAL answer back to the user.
    - This allows the agent to wrap up and send a natural-language response.
    - tools_used: list of names of tools that were used before arriving at the answer.
    """
    return {"answer": answer, "tools_used": tools_used}


# ==========================
# 2. Initialize LLM (Ollama Mistral model running locally)
# ==========================
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")


# ==========================
# 3. Define Prompt Template
# ==========================
# This tells the LLM how to behave:
# - It must use tools to compute answers when needed.
# - Scratchpad stores intermediate tool call results.
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
        # Placeholder to inject chat history dynamically
        MessagesPlaceholder(variable_name="chat_history"),
        # User’s actual query
        ("user", "{input}"),
        # Placeholder for agent scratchpad (tracks tool calls + outputs)
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# ==========================
# 4. Define Tools available to agent
# ==========================
tools = [final_answer, add, subtract, multiply, divide]

# ==========================
# 5. Create Runnable Graph
# ==========================
# This sets up the "pipeline" the agent follows:
# Input → Prompt → LLM + Tool Binding
agentic: RunnableSerializable = (
    {
        "input": lambda x: x["input"],  # Pass input text
        "chat_history": lambda x: x["chat_history"],  # Pass conversation history
        "agent_scratchpad": lambda x: x.get("agent_scratchpad", []),  # Track tool usage
    }
    | prompt  # Fill prompt template
    | llm.bind_tools(tools, tool_choice="any")  # Bind tools to LLM
)


# ==========================
# 6. First Run: Ask the LLM a Question
# ==========================
print("=== Runnable graph run ===")
tool_call = agentic.invoke({"input": "what is 10 + 10", "chat_history": []})

# tool_call contains the LLM’s plan → which tool to call + arguments
print("LLM suggested tool call:", tool_call)
print("Tool calls extracted:", tool_call.tool_calls)


# ==========================
# 7. Manually Execute the Suggested Tool
# ==========================
# Instead of letting LangChain run the tool, we show how to manually map tool names to functions
name2tool = {tool.name: tool.func for tool in tools}
print({"":name2tool})
# Actually execute the tool chosen by LLM
tool_out = name2tool[tool_call.tool_calls[0]["name"]](**tool_call.tool_calls[0]["args"])

# Wrap the tool execution result into a ToolMessage
tool_exec = ToolMessage(
    content=f"The {tool_call.tool_calls[0]['name']} tool returned {tool_out}",
    tool_call_id=tool_call.tool_calls[0]["id"],
)


# ==========================
# 8. Second Run: Give the LLM the Tool Output
# ==========================
# Now we feed back the tool result into the agent_scratchpad.
out = agentic.invoke(
    {
        "input": "What is 10 + 10",
        "chat_history": [],
        "agent_scratchpad": [tool_call, tool_exec],
    }
)

# Final output after using the tool
print("Final Answer:", out)
