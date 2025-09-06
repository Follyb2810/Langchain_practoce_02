from langchain.memory import ConversationSummaryMemory
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import ConversationChain
from langchain.schema import messages_from_dict, messages_to_dict

# #! ConversationChain` was deprecated in LangChain 0.2.7
# #! now use  ** RunnableWithMessageHistory **



# Initialize Ollama LLM
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# Setup conversation memory (summarizes the conversation)
memory = ConversationSummaryMemory(llm=llm, return_messages=True)

chain = ConversationChain(llm=llm, memory=memory, verbose=True)
# chain = RunnableWithMessageHistory(
#     llm=llm,  
#     memory=memory,  
#     verbose=True,  
# )

# Invoking the conversation step by step
# Each input is a dictionary with "input" key
chain.invoke({"input": "Hi, my name is Folly"})
chain.invoke({"input": "I'm researching the different types of conversational memory"})
chain.invoke(
    {
        "input": "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory"
    }
)
chain.invoke({"input": "Buffer memory just stores the entire conversation, right?"})
chain.invoke(
    {"input": "Buffer window memory stores the last k messages, dropping the rest"}
)
