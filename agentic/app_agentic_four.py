from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.messages import ToolMessage

# -------------------------------
# 1. Define Tools (math + final answer)
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
    """Provide a final natural language answer with list of tools used"""
    return {"answer": answer, "tools_used": tools_used}


# -------------------------------
# 2. LLM Setup
# -------------------------------

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# -------------------------------
# 3. Prompt Template
# -------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a helpful assistant. "
            "You MUST use the provided tools to answer user questions. "
            "Always call the correct tool with arguments, and finally call `final_answer`.",
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
"""
tool_choice="any" → LLM decides when to call tools.
tool_choice="required" → Forces the model to call some tool every time, even if not needed.
tool_choice="none" → Ignores tools completely.
tool_choice={"name": "add"} → Forces LLM to always call the add tool.

#? | llm.bind_tools(tools, tool_choice="any")  If you want to always use any your math tools,
#? | | llm.bind_tools(tools, tool_choice="required") If you want to always use your math tools,

"""
# -------------------------------
# 5. First Run → LLM should call a math tool
# -------------------------------

print("=== First Agent Run ===")
tool_call = agentic.invoke({"input": "what is 10 + 10", "chat_history": []})
print("Raw output:", tool_call)

if not tool_call.tool_calls:
    print("\n⚠️ LLM did not call any tool. Instead it said:")
    print(tool_call.content)
else:
    # Extract tool call
    tool_call_info = tool_call.tool_calls[0]
    print("\nTool call extracted:", tool_call_info)

    # Map tool name → actual function
    name2tool = {tool.name: tool.func for tool in tools}
    tool_out = name2tool[tool_call_info["name"]](**tool_call_info["args"])
    print("\nTool output:", tool_out)

    # Wrap result in ToolMessage
    tool_exec = ToolMessage(
        content=f"The {tool_call_info['name']} tool returned {tool_out}",
        tool_call_id=tool_call_info["id"],
    )

    # -------------------------------
    # 6. Second Run → LLM now sees tool result,
    #    and should call `final_answer`
    # -------------------------------
    out = agentic.invoke(
        {
            "input": "What is 10 + 10",
            "chat_history": [],
            "agent_scratchpad": [tool_call, tool_exec],
        }
    )

    print("\n=== Second Agent Run ===")
    print(out)

    # If it's calling final_answer
    if out.tool_calls and out.tool_calls[0]["name"] == "final_answer":
        fa_call = out.tool_calls[0]
        fa_out = name2tool[fa_call["name"]](**fa_call["args"])
        print("\n=== Final Answer Tool Output ===")
        print(fa_out)
