from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.config import ConfigurableFieldSpec


# Initialize Ollama LLM
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")


class ConversationSummaryMessageHistory(BaseChatMessageHistory):
    def __init__(self, llm: ChatOllama):
        # store messages and llm
        self.messages: list[BaseMessage] = []
        self.llm = llm

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages and update the summary."""
        self.messages.extend(messages)

        # build summarization prompt
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Given the existing conversation summary and the new messages, "
                "generate a new summary of the conversation. Ensure that all "
                "important information is preserved."
            ),
            HumanMessagePromptTemplate.from_template(
                "Existing conversation summary:\n{existing_summary}\n\n"
                "New messages:\n{messages}"
            )
        ])

        # get existing summary (if any)
        existing_summary = ""
        for m in self.messages:
            if isinstance(m, SystemMessage):
                existing_summary = m.content

        # format new messages as text
        new_messages_text = "\n".join([m.content for m in messages])

        # call LLM
        new_summary = self.llm.invoke(
            summary_prompt.format_messages(
                existing_summary=existing_summary,
                messages=new_messages_text,
            )
        )

        # replace with a single system summary message
        self.messages = [SystemMessage(content=new_summary.content)]

    def clear(self) -> None:
        """Clear history completely."""
        self.messages = []


# store conversation histories per session
chat_map = {}

def get_chat_history(session_id: str, llm: ChatOllama) -> ConversationSummaryMessageHistory:
    if session_id not in chat_map:
        chat_map[session_id] = ConversationSummaryMessageHistory(llm=llm)
    return chat_map[session_id]


# Example pipeline: here we just use the raw llm
# You could also wrap this with a prompt template or a chain
pipeline = llm


# Attach memory using RunnableWithMessageHistory
pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",     # input field key
    history_messages_key="history", # where history will be injected
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Session ID for tracking conversation",
            default="id_default",
        ),
        ConfigurableFieldSpec(
            id="llm",
            annotation=ChatOllama,
            name="LLM",
            description="LLM used for summarization",
            default=llm,
        )
    ]
)


# Invoke with session
pipeline_with_history.invoke(
    {"query": "Hi, my name is Josh"},
    config={"session_id": "id_123", "llm": llm}
)
