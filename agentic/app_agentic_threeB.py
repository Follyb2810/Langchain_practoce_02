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
def final_answer(answer: str, tools_used: list[str]) -> str:
    """Provide a final natural language answer with list of tools used"""
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
                "Do not respond directly unless explicitly instructed. "
                "Always call the correct tool with arguments. "
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

tools = [add, subtract, multiply, divide]

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
# 5. First Run (try to invoke a tool)
# -------------------------------

print("=== First Agent Run ===")
tool_call = agentic.invoke({"input": "what is 10 + 10", "chat_history": []})
print("Raw output:", tool_call)

# Guard clause to prevent crash
if not tool_call.tool_calls:
    print("\n⚠️ LLM did not call any tool. Instead it said:")
    print(tool_call.content)
else:
    # -------------------------------
    # 6. Execute the Tool
    # -------------------------------
    tool_call_info = tool_call.tool_calls[0]  # first tool call
    print("\nTool call extracted:", tool_call_info)

    # Map tool name → actual function
    name2tool = {tool.name: tool.func for tool in tools}
    print("\nDEBUG tool mapping:", name2tool)

    # Execute tool with unpacked args (**kwargs)
    tool_out = name2tool[tool_call_info["name"]](**tool_call_info["args"])
    print("\nTool output:", tool_out)

    # -------------------------------
    # 7. Send Tool Result Back to LLM
    # -------------------------------
    tool_exec = ToolMessage(
        content=f"The {tool_call_info['name']} tool returned {tool_out}",
        tool_call_id=tool_call_info["id"]
    )

    # Second round → give LLM the tool output in scratchpad
    out = agentic.invoke({
        "input": "What is 10 + 10",
        "chat_history": [],
        "agent_scratchpad": [tool_call, tool_exec]
    })

    print("\n=== Final Agent Answer ===")
    print(out.content)

