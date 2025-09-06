from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory

# LLM
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# SerpAPI tool
search = SerpAPIWrapper()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Useful for when you need to answer questions about current events or the web"
)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that uses SerpAPI when needed."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Build agent
agent = create_tool_calling_agent(llm=llm, tools=[search_tool], prompt=chat_template)

# Wrap in executor with memory
agent_executor = AgentExecutor(agent=agent, tools=[search_tool], memory=memory, verbose=True)

# Run
response = agent_executor.invoke({"input": "Who is the current president of France?"})
print(response)
