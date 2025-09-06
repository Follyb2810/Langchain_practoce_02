from langchain.memory import ConversationBufferMemory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_ollama import ChatOllama
from app_agentic_tool import add, divide, multiply, subtract
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
# Define memory
memory = ConversationBufferMemory(
    memory_key="chat_history",  # must match MessagesPlaceholder in prompt
    return_messages=True,  # return messages instead of raw text
)
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
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ),
    ]
)
tools = [add, subtract, multiply, divide]
# Create agent with tools and prompt
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

# Wrap with AgentExecutor (this one supports memory)
executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

print("=== AgentExecutor with memory ===")
print(executor.invoke({"input": "what is 10 + 10?"}))
print(executor.invoke({"input": "and then multiply that by 2"}))
