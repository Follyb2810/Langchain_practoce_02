# Core LangChain prompt/message utilities
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Ollama local LLM wrapper
from langchain_ollama import ChatOllama

# For returning final results
import json

# Math tools (custom)
from app_agentic_tool import add, divide, multiply, subtract, exponentiate


# -------------------------------
# Prompt template
# -------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        # System rules tell LLM how to behave with tools
        (
            "system",
            (
                "You're a helpful assistant. Always try a tool first. "
                "If tool results appear in the scratchpad, use them to answer "
                "the user directly instead of calling more tools."
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Tools available to the LLM
tools = [add, subtract, multiply, exponentiate]

# Local Ollama LLM (Mistral model)
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# -------------------------------
# Simple agent pipeline (prompt → LLM with tools)
# -------------------------------
agent: RunnableSerializable = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "agent_scratchpad": lambda x: x.get("agent_scratchpad", []),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)

# Example run: "What is 10 + 10"
tool_call = agent.invoke({"input": "What is 10 + 10", "chat_history": []})

# Map tool names back to Python functions
name2tool = {tool.name: tool.func for tool in tools}

# Execute the first tool call
tool_exec_content = name2tool[tool_call.tool_calls[0]["name"]](
    **tool_call.tool_calls[0]["args"]
)


# -------------------------------
# Custom Agent Executor
# -------------------------------
class CustomAgentExecutor:
    chat_history: list[BaseMessage]

    def __init__(self, max_iterations: int = 3):
        self.chat_history = []
        self.max_iterations = max_iterations
        # Build an agent internally (same structure as above)
        self.agent: RunnableSerializable = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", []),
            }
            | prompt
            | llm.bind_tools(tools, tool_choice="any")
        )

    def invoke(self, input: str) -> dict:
        """Run agent iteratively until final answer or iteration limit."""
        count = 0
        agent_scratchpad = []

        while count < self.max_iterations:
            # Ask LLM what to do (likely a tool call)
            tool_call = self.agent.invoke(
                {
                    "input": input,
                    "chat_history": self.chat_history,
                    "agent_scratchpad": agent_scratchpad,
                }
            )
            agent_scratchpad.append(tool_call)

            # Extract tool request
            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_call_id = tool_call.tool_calls[0]["id"]

            # Execute tool
            tool_out = name2tool[tool_name](**tool_args)

            # Feed tool result back into scratchpad
            tool_exec = ToolMessage(content=f"{tool_out}", tool_call_id=tool_call_id)
            agent_scratchpad.append(tool_exec)

            # Debug trace
            print(f"{count}: {tool_name}({tool_args})")
            count += 1

            # Stop if final answer signaled
            if tool_name == "final_answer":
                break

        # Persist final exchange into chat history
        final_answer = tool_out["answer"]
        self.chat_history.extend(
            [HumanMessage(content=input), AIMessage(content=final_answer)]
        )

        return json.dumps(tool_out)
