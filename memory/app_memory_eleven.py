from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferWindowMemory


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
# 3. Setup session memory map
# -------------------------------
chat_map = {}

def get_chat_memory(session_id: str, k: int = 3):
    """
    Return a ConversationBufferWindowMemory for a session.
    Keeps only the last `k` messages.
    """
    if session_id not in chat_map:
        memory = ConversationBufferWindowMemory(
            k=k, return_messages=True, memory_key="history"
        )

        # Optional: preload initial messages
        memory.save_context(
            {"input": "Hi, my name is Folly"},
            {"output": "Hey Folly, what's up? I'm an AI model called Folyb."},
        )
        memory.save_context(
            {"input": "I'm researching the different types of conversational memory"},
            {"output": "That's interesting, what are some examples?"},
        )

        chat_map[session_id] = memory

    return chat_map[session_id].chat_memory  # return underlying message list


# -------------------------------
# 4. Wrap pipeline with memory
# -------------------------------
pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_memory,
    input_messages_key="query",
    history_messages_key="history",
)

# -------------------------------
# 5. Run session
# -------------------------------
session_id = "user123"

response1 = pipeline_with_history.invoke(
    {"query": "What is my name again?"},
    config={"configurable": {"session_id": session_id}},
)

response2 = pipeline_with_history.invoke(
    {"query": "Hi, my name is folly"},
    config={"configurable": {"session_id": session_id}},
)

response3 = pipeline_with_history.invoke(
    {"query": "What was my first question?"},
    config={"configurable": {"session_id": session_id}},
)

print("Bot:", response1)
print("Bot:", response2)
print("Bot:", response3)
