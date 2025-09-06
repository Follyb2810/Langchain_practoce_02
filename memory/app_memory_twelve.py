from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# Full conversation memory
memory = ConversationBufferMemory(return_messages=True)

# Preload messages
memory.save_context({"input": "Hi, my name is Folly"}, {"output": "Hello Folly!"})
memory.save_context({"input": "How are you?"}, {"output": "I'm an AI, I'm good!"})

# Access memory
print(memory.load_memory_variables({}))

###?


# Windowed memory (keep last 3 messages)
window_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Add messages
window_memory.save_context({"input": "Hi"}, {"output": "Hello!"})
window_memory.save_context({"input": "How are you?"}, {"output": "I'm good"})
window_memory.save_context(
    {"input": "What's AI?"}, {"output": "AI is artificial intelligence"}
)
window_memory.save_context(
    {"input": "Tell me a joke"}, {"output": "Why did the robot cross the road?"}
)

# Only last 3 messages remain
print(window_memory.load_memory_variables({}))


#? Create chat history
history = InMemoryChatMessageHistory()

# Add messages
history.add_user_message("Hi, my name is Folly")
history.add_ai_message("Hello Folly!")
history.add_user_message("Can you remind me of our last chat?")
history.add_ai_message("Sure! Last time we discussed conversational memory.")

# Access messages manually
for msg in history.messages:
    print(msg.type, msg.content)
