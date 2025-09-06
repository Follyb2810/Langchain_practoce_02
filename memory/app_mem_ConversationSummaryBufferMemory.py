from langchain_core.runnables import ConfigurableFieldSpec
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory


llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

system_prompt = "You are a helpful assistant called Zeta."

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)


class ConversationSummaryBufferMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    llm: ChatOpenAI = Field(default_factory=ChatOpenAI)
    k: int = Field(default_factory=int)

    def __init__(self, llm: ChatOpenAI, k: int):
        super().__init__(llm=llm, k=k)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, removing any messages beyond
        the last `k` messages and summarizing the messages that we
        drop.
        """
        existing_summary: SystemMessage | None = None
        old_messages: list[BaseMessage] | None = None
        # see if we already have a summary message
        if len(self.messages) > 0 and isinstance(self.messages[0], SystemMessage):
            print(">> Found existing summary")
            existing_summary = self.messages.pop(0)
        # add the new messages to the history
        self.messages.extend(messages)
        # check if we have too many messages
        if len(self.messages) > self.k:
            print(
                f">> Found {len(self.messages)} messages, dropping "
                f"oldest {len(self.messages) - self.k} messages."
            )
            # pull out the oldest messages...
            old_messages = self.messages[: self.k]
            # ...and keep only the most recent messages
            self.messages = self.messages[-self.k :]
        if old_messages is None:
            print(">> No old messages to update summary with")
            # if we have no old_messages, we have nothing to update in summary
            return
        # construct the summary chat messages
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "Given the existing conversation summary and the new messages, "
                    "generate a new summary of the conversation. Ensuring to maintain "
                    "as much relevant information as possible."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Existing conversation summary:\n{existing_summary}\n\n"
                    "New messages:\n{old_messages}"
                ),
            ]
        )
        # format the messages and invoke the LLM
        new_summary = self.llm.invoke(
            summary_prompt.format_messages(
                existing_summary=existing_summary, old_messages=old_messages
            )
        )
        print(f">> New summary: {new_summary.content}")
        # prepend the new summary to the history
        self.messages = [SystemMessage(content=new_summary.content)] + self.messages

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []


chat_map = {}


def get_chat_history(
    session_id: str, llm: ChatOpenAI, k: int
) -> ConversationSummaryBufferMessageHistory:
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = ConversationSummaryBufferMessageHistory(llm=llm, k=k)
    # return the chat history
    return chat_map[session_id]


pipeline = prompt_template | llm

pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID to use for the chat history",
            default="id_default",
        ),
        ConfigurableFieldSpec(
            id="llm",
            annotation=ChatOpenAI,
            name="LLM",
            description="The LLM to use for the conversation summary",
            default=llm,
        ),
        ConfigurableFieldSpec(
            id="k",
            annotation=int,
            name="k",
            description="The number of messages to keep in the history",
            default=4,
        ),
    ],
)


pipeline_with_history.invoke(
    {"query": "Hi, my name is James"},
    config={"session_id": "id_123", "llm": llm, "k": 4},
)
chat_map["id_123"].messages
