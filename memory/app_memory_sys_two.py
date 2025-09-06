from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Custom conversation summary class
class ConversationSummary(BaseModel):
    # Store messages as a list of BaseMessage (System/Human/AI)
    messages: list[BaseMessage] = Field(default_factory=list)

    # Store the LLM instance
    llm: ChatOllama = Field(default_factory=lambda: ChatOllama(model="mistral", base_url="http://localhost:11434"))

    def __init__(self, llm: ChatOllama, **kwargs):
        # fix typo: was "selt", should be "self"
        super().__init__(llm=llm, **kwargs)

    def add_message(self, message: list[BaseMessage]) -> None:
        # Extend the conversation with the new incoming messages
        self.messages.extend(message)

        # Build a summarization prompt
        # SystemMessagePromptTemplate: defines instruction for summarization
        # HumanMessagePromptTemplate: provides current summary + new messages
        summary_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "You are a helpful assistant that maintains a running summary of the conversation."
                    " Update the summary by combining the existing summary with the new message(s)."
                    " Ensure no important detail is lost."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Existing conversation summary:\n{existing_summary}\n\n"
                    "New Messages:\n{messages}"
                ),
            ]
        )

        # Call the LLM to produce a new summary
        new_summary = self.llm.invoke(
            summary_template.format_messages(
                existing_summary="\n".join([m.content for m in self.messages if isinstance(m, SystemMessage)]),
                messages="\n".join([m.content for m in message]),
            )
        )

        # Replace stored messages with the updated summary
        # Store summary as a SystemMessage because it represents *context*, not user input
        self.messages = [SystemMessage(content=new_summary.content)]
