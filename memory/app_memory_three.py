from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory,BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.messages import messages_to_dict, messages_from_dict


class MongoChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, collection, session_id):
        self.collection = collection
        self.session_id = session_id

    def add_message(self, message):
        self.collection.update_one(
            {"session_id": self.session_id},
            {"$push": {"messages": messages_to_dict([message])}},
            upsert=True,
        )

    @property
    def messages(self):
        doc = self.collection.find_one({"session_id": self.session_id})
        return messages_from_dict(doc["messages"]) if doc else []


# Initialize LLM
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# System prompt
sys_prompt = "You are a helpful assistant called Follyb."

# Prompt template with history
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(sys_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

# Pipeline
pipeline = prompt_template | llm | StrOutputParser()

# Chat session store
chat_map = {}


def get_chat_memory(session_id: str) ->InMemoryChatMessageHistory:
    if session_id not in chat_map:
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]


# Runnable with history
pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_memory,
    input_messages_key="query",
    history_messages_key="history",
)

# Example usage
session_id = "user123"

response1 = pipeline_with_history.invoke(
    {"query": "Hi, my name is Folly"},
    config={"configurable": {"session_id": session_id}},
)
print("Bot:", response1)

response2 = pipeline_with_history.invoke(
    {"query": "Can you remind me who attended to me last time?"},
    config={"configurable": {"session_id": session_id}},
)
print("Bot:", response2)
