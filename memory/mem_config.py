from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.config import ConfigurableFieldSpec


# --- Custom Chat History that requires BOTH session_id and llm ---
class SimpleHistory(BaseChatMessageHistory):
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.messages: list[BaseMessage] = []

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


# --- Function to create/fetch a chat history ---
chat_map = {}
def get_chat_history(session_id: str, llm: ChatOllama) -> SimpleHistory:
    if session_id not in chat_map:
        chat_map[session_id] = SimpleHistory(llm=llm)
    return chat_map[session_id]


# --- Base LLM pipeline ---
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")


# --- Wrap pipeline with history + config spec ---
pipeline_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        # 1. Required field: session_id
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Unique ID for the conversation session"
        ),
        # 2. Optional field: llm (with default)
        ConfigurableFieldSpec(
            id="llm",
            annotation=ChatOllama,
            name="LLM",
            description="LLM instance used for this session",
            default=llm,
        ),
    ]
)


# --- Usage ---
response = pipeline_with_history.invoke(
    {"query": "Hello, who are you?"},
    config={"session_id": "chat_1", "llm": llm}   # ✅ pass args defined above
)

print(response)
