from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables.base import RunnableSerializable
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json


from app_agentic_tool import add, divide, multiply, subtract, exponentiate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You're a helpful assistant. When answering a user's question "
                "you should first use one of the tools provided. After using a "
                "tool, the tool output will be provided in the 'scratchpad'. "
                "If the scratchpad has an answer, do NOT use more tools—just answer directly."
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [add, subtract, multiply, divide, exponentiate]

name2tool = {tool.name: tool.func for tool in tools}

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")


class CustomAgentExecutor:
    """A simple agent executor that can use tools iteratively
    until it produces a final answer or hits max_iterations."""

    def __init__(self, max_iterations: int = 3):
        self.chat_history: list[BaseMessage] = []
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

    def invoke(self, input: str) -> dict:
        """Run the agent loop for one user input."""
        count = 0
        agent_scratchpad = []
        final_answer = None

        while count < self.max_iterations:
            tool_call = self.agent.invoke(
                {
                    "input": input,
                    "chat_history": self.chat_history,
                    "agent_scratchpad": agent_scratchpad,
                }
            )

            if not tool_call.tool_calls:
                final_answer = tool_call.content
                break

            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_call_id = tool_call.tool_calls[0]["id"]

            tool_out = name2tool[tool_name](**tool_args)

            tool_exec = ToolMessage(content=f"{tool_out}", tool_call_id=tool_call_id)
            agent_scratchpad.append(tool_exec)

            print(f"Iteration {count}: {tool_name}({tool_args}) = {tool_out}")

            if isinstance(tool_out, dict) and "answer" in tool_out:
                final_answer = tool_out["answer"]
                break

            count += 1

        if final_answer:
            self.chat_history.extend(
                [HumanMessage(content=input), AIMessage(content=final_answer)]
            )
            return {"answer": final_answer}
        else:
            return {"answer": "No valid final answer after iterations."}


if __name__ == "__main__":
    executor = CustomAgentExecutor(max_iterations=3)

    print(executor.invoke("What is 10 + 10?"))
    print(executor.invoke("What is (2 + 3) * 4?"))
