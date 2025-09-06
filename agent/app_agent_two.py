from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.agents import tool_calling_agent
from app_agent_one import add, exponentiate, multiply, subtract

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Build a chat prompt template. This is a "blueprint" that defines:
# - system instructions
# - where to insert history
# - where to insert the latest user input
# - where to insert intermediate reasoning (agent_scratchpad)
chat_template = ChatPromptTemplate.from_messages(
    [
        # 1. A system message (fixed instruction to guide the model's behavior)
        ("system", "you are a helpful assistant"),
        # 2. Placeholder for prior conversation turns (chat_history is a list of past messages)
        MessagesPlaceholder(variable_name="chat_history"),
        # 3. The current human/user input (runtime variable {input})
        ("human", "{input}"),
        # 4. Placeholder for agent's scratchpad
        #    This is where intermediate reasoning steps (e.g., tool calls, thoughts)
        #    are injected during Agent execution.
        ("placeholder", "{agent_scratchpad}"),
    ]
)
tools = [add, subtract, multiply, exponentiate]
agent = tool_calling_agent(llm=llm, tools=tools, prompt=chat_template)

result =agent.invoke(
    {
        "input": "what is 10.07 multiplied by 7.687",
        "chat_history": memory.chat_memory.messages,
        "intermediate_step": [],
    }
)

print(result)
