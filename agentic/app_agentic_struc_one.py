from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.messages import ToolMessage

# -------------------------------
# 1. Define Tools (simple math ops)
# -------------------------------

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
def final_answer(answer: str, tools_used: list[str]) -> dict:
    """Return the final structured JSON answer"""
    return {"answer": answer, "tools_used": tools_used}


# -------------------------------
# 2. LLM Setup
# -------------------------------

llm = ChatOllama(
    model="mistral",
    base_url="http://localhost:11434"
)

# -------------------------------
# 3. Prompt Template
# -------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You're a helpful assistant. "
                "You MUST use the provided tools to answer user questions. "
                "Always call the correct tool with arguments. "
                "When you have the final result, ALWAYS call `final_answer` "
                "with JSON fields: { 'answer': string, 'tools_used': [list of tool names] }. "
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# -------------------------------
# 4. Agent Graph
# -------------------------------

tools = [add, subtract, multiply, divide, final_answer]

agentic: RunnableSerializable = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "agent_scratchpad": lambda x: x.get("agent_scratchpad", []),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)

# -------------------------------
# 5. Run Example
# -------------------------------

print("=== First Agent Run ===")
tool_call = agentic.invoke({"input": "what is 10 + 10", "chat_history": []})
print("Raw output:", tool_call)

if not tool_call.tool_calls:
    print("\n⚠️ LLM did not call any tool. Instead it said:")
    print(tool_call.content)
else:
    # Pick first tool call
    tool_call_info = tool_call.tool_calls[0]
    print("\nTool call extracted:", tool_call_info)

    # Map tool name → function
    name2tool = {tool.name: tool.func for tool in tools}

    # Execute tool
    tool_out = name2tool[tool_call_info["name"]](**tool_call_info["args"])
    print("\nTool output:", tool_out)

    # Send back to LLM
    tool_exec = ToolMessage(
        content=str(tool_out),  # structured tool result
        tool_call_id=tool_call_info["id"]
    )

    # Final round
    out = agentic.invoke({
        "input": "what is 10 + 10",
        "chat_history": [],
        "agent_scratchpad": [tool_call, tool_exec]
    })

    print("\n=== Final Agent JSON Answer ===")
    print(out.content)
