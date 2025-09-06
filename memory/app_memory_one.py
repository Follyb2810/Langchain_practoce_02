from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain 
#! ConversationChain` was deprecated in LangChain 0.2.7
#! now use  ** RunnableWithMessageHistory **

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

memory = ConversationBufferMemory(return_messages=True)

memory.save_context(
    {"input": "Hi, my name is Folly"},
    {"output": "Hey Folly, what's up? I'm an AI model called Folyb."},
)
memory.save_context(
    {"input": "I'm researching the different types of conversational memory"},
    {"output": "That's interesting, what are some examples?"},
)
memory.save_context(
    {
        "input": "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory"
    },
    {"output": "That's interesting, what's the difference?"},
)
memory.save_context(
    {"input": "Buffer memory just stores the entire conversation, right?"},
    {"output": "That makes sense, what about ConversationBufferWindowMemory?"},
)
memory.save_context(
    {"input": "Buffer window memory stores the last k messages, dropping the rest"},
    {"output": "Very cool!"},
)
memory.chat_memory.add_user_message(
    "So which one should I use if I want to keep the entire history?"
)
memory.chat_memory.add_ai_message(
    "You should use ConversationBufferMemory for the full history. "
    "If you only want the last k turns, use ConversationBufferWindowMemory."
)
memory.chat_memory.add_user_message(
    "Can you give me a quick code example of BufferWindowMemory?"
)
memory.chat_memory.add_ai_message(
    "Sure! You can initialize it with something like:\n\n"
    "```python\n"
    "from langchain.memory import ConversationBufferWindowMemory\n"
    "memory = ConversationBufferWindowMemory(k=3, return_messages=True)\n"
    "```"
)
print("Conversation history (via load_memory_variables):")
print(memory.load_memory_variables({}))


print("\nUpdated conversation history:")
for msg in memory.chat_memory.messages:
    role = "User" if msg.type == "human" else "AI"
    print(f"{role}: {msg.content}")

chain = ConversationChain(llm=llm, memory=memory, verbose=True)

response = chain.predict(input="Can you remind me what BufferWindowMemory does?")
print("\nChain response:")
print(response)
