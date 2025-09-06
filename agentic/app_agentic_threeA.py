# ===========================================
# 0. Import necessary modules from LangChain + Ollama integration
# ===========================================
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.messages import ToolMessage

# ===========================================
# 1. Define Tools
# ===========================================

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
    """
    Provide the FINAL answer back to the user.
    This allows the agent to wrap up and send a natural-language response.
    """
    return {"answer": answer, "tools_used": tools_used}


# ===========================================
# 2. Initialize LLM (Ollama Mistral model running locally)
# ===========================================
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")


# ===========================================
# 3. Define Prompt Template
# ===========================================
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
        MessagesPlaceholder(variable_name="chat_history"),  # previous messages
        ("user", "{input}"),  # current user query
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # tool traces
    ]
)


# ===========================================
# 4. Define Tools available to agent
# ===========================================
tools = [final_answer, add, subtract, multiply, divide]


# ===========================================
# 5. Create Runnable Graph
# ===========================================
agentic: RunnableSerializable = (
    {
        "input": lambda x: x["input"],  
        "chat_history": lambda x: x["chat_history"],  
        "agent_scratchpad": lambda x: x.get("agent_scratchpad", []),  
    }
    | prompt  
    | llm.bind_tools(tools, tool_choice="any")  
)


# ===========================================
# 6. First Run: Ask the LLM a Question
# ===========================================
print("=== Runnable graph run ===")
tool_call = agentic.invoke({"input": "what is 10 + 10", "chat_history": []})

print("LLM suggested tool call:", tool_call)
print("Tool calls extracted:", tool_call.tool_calls)


# ===========================================
# 7. Manually Execute the Suggested Tool
# ===========================================

# 1. Map tool names to functions
name2tool = {t.name: t.func for t in tools}
print("DEBUG: Tool mapping =>", name2tool, "\n")

# 2. Extract tool call info
tool_call_info = tool_call.tool_calls[0]
print("DEBUG: Tool call returned by LLM =>", tool_call_info, "\n")

# 3. Extract details
tool_name = tool_call_info["name"]  # e.g. "add"
tool_args = tool_call_info["args"]  # e.g. {"x": 10, "y": 10}

print(f"DEBUG: Tool chosen => {tool_name}")
print(f"DEBUG: Args => {tool_args}\n")

# 4. Lookup and call the tool
selected_tool = name2tool[tool_name]
print(f"DEBUG: Python function selected => {selected_tool}\n")

print(f"DEBUG: About to call => {tool_name}(**{tool_args})")
tool_out = selected_tool(**tool_args)
print("DEBUG: Tool execution result =>", tool_out, "\n")

# 5. Wrap result as ToolMessage
tool_exec = ToolMessage(
    content=f"The {tool_name} tool returned {tool_out}",
    tool_call_id=tool_call_info["id"],
)


# ===========================================
# 8. Second Run: Feed Tool Output Back
# ===========================================
out = agentic.invoke(
    {
        "input": "What is 10 + 10",
        "chat_history": [],
        "agent_scratchpad": [tool_call, tool_exec],
    }
)

print("=== Final Answer ===")
print(out)
