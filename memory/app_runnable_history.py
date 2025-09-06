from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

sys_prompt = "You are a helpful assistant called Follyb."

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

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(sys_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

query = "How many keystrokes are needed to type the numbers from 1 to 500?"

pipeline = (
    # RunnableMap({
    #     "query": lambda x: x["query"],
    #     "history": lambda x: memory.chat_memory.messages
    # })
    # |
    prompt_template
    | llm
    | StrOutputParser()
)


chat_map = {}


def get_chat_memory(session_id: str):
    if session_id not in chat_map:
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]


pipeline_with_history = RunnableWithMessageHistory(
    pipeline, get_session_history=get_chat_memory, input_messages_key="query",
    history_messages_key='history'
)
