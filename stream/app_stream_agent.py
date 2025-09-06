from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langchain_ollama import ChatOllama
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables import ConfigurableField
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import asyncio
from langchain.callbacks.base import AsyncCallbackHandler


# LLM with Ollama backend
llm = ChatOllama(
    model="mistral", base_url="http://localhost:11434"
).configurable_fields(
    callbacks=ConfigurableField(
        id="callbacks",
        name="callbacks",
        description="A list of callbacks to use for streaming",
    )
)


# === Tools ===
@tool
def add(x: float, y: float) -> float:
    """Add two numbers together and return the result."""
    return x + y


@tool
def subtract(x: float, y: float) -> float:
    """Subtract y from x and return the result."""
    return x - y


@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers and return the result."""
    return x * y


@tool
def divide(x: float, y: float) -> float:
    """Divide x by y and return the result. Raises error if y = 0."""
    if y == 0:
        raise ValueError("Division by zero is not allowed.")
    return x / y


@tool
def exponentiate(x: float, y: float) -> float:
    """Raise x to the power of y and return the result."""
    return x**y


@tool
def final_answer(answer: str, tools_used: list[str]) -> dict:
    """Provide a final answer to the user.
    The answer should be in natural language as this will be provided
    directly to the user. The tools_used must include a list of tool
    names that were used within the `scratchpad`.
    """
    return {"answer": answer, "tools_used": tools_used}


# === Prompt Template ===
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
        {"user": "{input}"},
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [add, multiply, exponentiate, divide, final_answer]


# === Runnable agent (manual tool binding) ===
agentic: RunnableSerializable = (
    {
        "input": lambda x: x["input"],  # extract user input
        "chat_history": lambda x: x["chat_history"],  # pass history
        "agent_scratchpad": lambda x: x.get("agent_scratchpad", []),  # pass scratchpad
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")  # bind tools manually
)


print("=== Runnable graph run ===")
tool_call = agentic.invoke({"input": "what is 10 + 10", "chat_history": []})
print(tool_call)


# === Tool lookup ===
name2tool = {tool.name: tool.func for tool in tools}


# === Custom Agent Executor ===
class CustomAgentExecutor:
    chat_history: list[BaseMessage]

    def __init__(self, max_iterations: int = 3):
        self.chat_history = []
        self.max_iterations = max_iterations
        self.agent: RunnableSerializable = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", []),
            }
            | prompt
            | llm.bind_tools(tools, tool_choice="any")
        )

    async def invoke(self, input: str, callback_handler=None, verbose=False) -> dict:
        """Run the agent iteratively until a final_answer tool call is made."""
        count = 0
        agent_scratchpad = []

        while count < self.max_iterations:
            # step 1: ask the LLM for next tool call
            out = self.agent.invoke(
                {
                    "input": input,
                    "chat_history": self.chat_history,
                    "agent_scratchpad": agent_scratchpad,
                }
            )

            # step 2: if it's final_answer → stop
            if out.tool_calls[0]["name"] == "final_answer":
                break

            # step 3: otherwise execute tool
            agent_scratchpad.append(out)  # add tool call to scratchpad
            tool_out = name2tool[out.tool_calls[0]["name"]](**out.tool_calls[0]["args"])
            action_str = f"The {out.tool_calls[0]['name']} tool returned {tool_out}"

            agent_scratchpad.append(
                {
                    "role": "tool",
                    "content": action_str,
                    "tool_call_id": out.tool_calls[0]["id"],
                }
            )

            if verbose:
                print(f"{count}: {action_str}")

            count += 1

        # Finalize
        final_answer = out.tool_calls[0]["args"]
        final_answer_str = json.dumps(final_answer)

        self.chat_history.append({"input": input, "output": final_answer_str})
        self.chat_history.extend(
            [HumanMessage(content=input), AIMessage(content=final_answer_str)]
        )

        return final_answer


agent_executor = CustomAgentExecutor()


# === Callback handler for async streaming ===
class QueueCallbackHandler(AsyncCallbackHandler):
    """Callback handler that puts tokens into a queue."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.final_answer_seen = False

    async def __aiter__(self):
        while True:
            token_or_done = await self.queue.get()
            if token_or_done == "<<DONE>>":
                return
            if token_or_done:
                yield token_or_done

    async def on_llm_new_token(self, *args, **kwargs) -> None:
        """Put new token in the queue."""
        chunk = kwargs.get("chunk")
        if chunk:
            if tool_calls := chunk.message.additional_kwargs.get("tool_calls"):
                if tool_calls[0]["function"]["name"] == "final_answer":
                    self.final_answer_seen = True
        await self.queue.put(chunk)

    async def on_llm_end(self, *args, **kwargs) -> None:
        """Mark completion in the queue."""
        if self.final_answer_seen:
            await self.queue.put("<<DONE>>")
        else:
            await self.queue.put("<<STEP_END>>")


# === Example streaming run ===
async def main():
    queue = asyncio.Queue()
    streamer = QueueCallbackHandler(queue)

    task = asyncio.create_task(agent_executor.invoke("What is 10 + 10", streamer, verbose=True))

    async for token in streamer:
        if token == "<<STEP_END>>":
            print("\n", flush=True)
        elif tool_calls := token.message.additional_kwargs.get("tool_calls"):
            if tool_name := tool_calls[0]["function"]["name"]:
                print(f"Calling {tool_name}...", flush=True)
            if tool_args := tool_calls[0]["function"]["arguments"]:
                print(f"{tool_args}", end="", flush=True)

    await task


# Run the async loop
asyncio.run(main())
