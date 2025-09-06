from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.chat_history import InMemoryChatMessageHistory

# -------------------------------
# 1. Setup LLM
# -------------------------------
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
sys_prompt = "You are a helpful assistant called Follyb."

# -------------------------------
# 2. Prompt template
# -------------------------------
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(sys_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)
pipeline = prompt_template | llm | StrOutputParser()

# -------------------------------
# 3. Setup different memory types
# -------------------------------

# Full conversation memory
full_memory = ConversationBufferMemory(return_messages=True)
full_memory.save_context({"input": "Hi, I'm Folly"}, {"output": "Hello Folly!"})
full_memory.save_context({"input": "How are you?"}, {"output": "I'm good, thank you!"})

# Window memory (last 2 messages only)
window_memory = ConversationBufferWindowMemory(k=2, return_messages=True)
window_memory.save_context({"input": "Hi"}, {"output": "Hello!"})
window_memory.save_context({"input": "What's AI?"}, {"output": "Artificial intelligence!"})
window_memory.save_context({"input": "Tell me a joke"}, {"output": "Why did the robot cross the road?"})

# In-memory chat history
inmemory_history = InMemoryChatMessageHistory()
inmemory_history.add_user_message("Hi, my name is Folly")
inmemory_history.add_ai_message("Hello Folly!")
inmemory_history.add_user_message("Can you remind me of our last chat?")
inmemory_history.add_ai_message("Sure! Last time we discussed conversational memory.")

# -------------------------------
# 4. Memory map for RunnableWithMessageHistory
# -------------------------------
chat_map = {
    "full": full_memory.chat_memory,
    "window": window_memory.chat_memory,
    "inmemory": inmemory_history,
}


def get_chat_memory(session_id: str):
    # Return the appropriate chat memory
    return chat_map[session_id]


pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_memory,
    input_messages_key="query",
    history_messages_key="history",
)

# -------------------------------
# 5. Run with different memories
# -------------------------------

for mem_type in ["full", "window", "inmemory"]:
    print(f"\n--- Using memory type: {mem_type} ---")
    response = pipeline_with_history.invoke(
        {"query": "What did we talk about before?"},
        config={"configurable": {"session_id": mem_type}},
    )
    print("Bot:", response)
