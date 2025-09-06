from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from app_agent_one import add, exponentiate, multiply, subtract

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

tools = [add, subtract, multiply, exponentiate]

# Create tool-calling agent
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=chat_template)

# Wrap in an executor so memory works automatically
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# Run agent
response = agent_executor.invoke(
    {"input": "what is 10.07 multiplied by 7.687", "chat_history": memory}
)
print(response)
print(memory.chat_memory.messages)
responseA = agent_executor.invoke({"input": "my name is folly", "chat_history": memory})
print(responseA)
print(memory.chat_memory.messages)
responseB = agent_executor.invoke({"input":'what is 9 plus 10,minus 4 * 2,to the power of 3','chat_history':memory})
responseC = agent_executor.invoke({"input": "What is my name", "chat_history": memory})
print(responseC)