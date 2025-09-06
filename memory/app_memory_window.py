from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

memory = ConversationBufferWindowMemory(k=4, return_messages=True)
memory.chat_memory.add_user_message("Hi, my name is Josh")
memory.chat_memory.add_ai_message("Hey Josh, what's up? I'm an AI model called Zeta.")
memory.chat_memory.add_user_message("I'm researching the different types of conversational memory.")
memory.chat_memory.add_ai_message("That's interesting, what are some examples?")
memory.chat_memory.add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
memory.chat_memory.add_ai_message("That's interesting, what's the difference?")
memory.chat_memory.add_user_message("Buffer memory just stores the entire conversation, right?")
memory.chat_memory.add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
memory.chat_memory.add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
memory.chat_memory.add_ai_message("Very cool!")

chain = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=True
)
chain.invoke({"input": "what is my name again?"})
