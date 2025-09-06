from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


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
# 3. Prepare chat map + preload history outside
# -------------------------------
chat_map = {}

# preload history once
initial_history = InMemoryChatMessageHistory()
initial_history.add_user_message("Hi, my name is Folly")
initial_history.add_ai_message("Hey Folly, what's up? I'm an AI model called Folyb.")
initial_history.add_user_message("I'm researching the different types of conversational memory")
initial_history.add_ai_message("That's interesting, what are some examples?")
initial_history.add_user_message(
    "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory"
)
initial_history.add_ai_message("That's interesting, what's the difference?")
initial_history.add_user_message("Buffer memory just stores the entire conversation, right?")
initial_history.add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
initial_history.add_user_message(
    "Buffer window memory stores the last k messages, dropping the rest"
)
initial_history.add_ai_message("Very cool!")
initial_history.add_user_message("So which one should I use if I want to keep the entire history?")
initial_history.add_ai_message(
    "You should use ConversationBufferMemory for the full history. "
    "If you only want the last k turns, use ConversationBufferWindowMemory."
)
initial_history.add_user_message("Can you give me a quick code example of BufferWindowMemory?")
initial_history.add_ai_message(
    "Sure! You can initialize it with something like:\n\n"
    "```python\n"
    "from langchain.memory import ConversationBufferWindowMemory\n"
    "memory = ConversationBufferWindowMemory(k=3, return_messages=True)\n"
    "```"
)


def get_chat_memory(session_id: str):
    if session_id not in chat_map:
        # copy the initial history for each new session
        chat_map[session_id] = InMemoryChatMessageHistory()
        chat_map[session_id].messages = list(initial_history.messages)
    return chat_map[session_id]


pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_memory,
    input_messages_key="query",
    history_messages_key="history",
)

# -------------------------------
# 4. Run with session
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
    {"query": "What is my name again?"},
    config={"configurable": {"session_id": session_id}},
)

print("Bot:", response1)
print("Bot:", response2)
print("Bot:", response3)
print("Bot:", chat_map[session_id])
