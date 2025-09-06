from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_ollama import ChatOllama


class ConversationSummary:
    def __init__(self, llm: ChatOllama):
        # keep track of all messages (system, human, ai)
        self.messages: list[BaseMessage] = []
        # store llm instance
        self.llm = llm

    def add_message(self, message: list[BaseMessage]) -> None:
        # append new messages to history
        self.messages.extend(message)

        # build summarization prompt
        summary_template = ChatPromptTemplate.from_messages(
            [
                # instruction to LLM (system-level role, not user)
                SystemMessagePromptTemplate.from_template(
                    "Given the existing conversation summary and the new message(s), "
                    "generate an updated summary of the conversation. "
                    "Ensure to keep all important details."
                ),
                # supply the old summary + new content
                HumanMessagePromptTemplate.from_template(
                    "Existing summary:\n{existing_summary}\n\n"
                    "New messages:\n{messages}"
                ),
            ]
        )

        # extract old summary if it exists (stored as SystemMessage)
        existing_summary = ""
        for m in self.messages:
            if isinstance(m, SystemMessage):
                existing_summary = m.content

        # call LLM to produce new summary
        new_summary = self.llm.invoke(
            summary_template.format_messages(
                existing_summary=existing_summary,
                messages="\n".join([m.content for m in message]),
            )
        )

        # overwrite stored messages with the new summary (as SystemMessage)
        # reason: summary is not user or ai content → it's context, so SystemMessage is correct
        self.messages = [SystemMessage(content=new_summary.content)]
