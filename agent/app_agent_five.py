from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationKGMemory
)
from langchain.chains import LLMChain

# === LLM ===
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# === Prompt Template ===
# This template is reused across all memory types.
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),  # memory goes here
        ("human", "{input}")
    ]
)


# ======================================================
# 1. ConversationBufferMemory (keeps the ENTIRE history)
# ======================================================
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)
buffer_chain = LLMChain(llm=llm, prompt=chat_template, memory=buffer_memory)

print("=== ConversationBufferMemory ===")
print(buffer_chain.run(input="Hello, who won the 2018 World Cup?"))
print(buffer_chain.run(input="Who was the captain of that team?"))
# Memory now has BOTH questions + the model’s answers in full.


# ======================================================
# 2. ConversationBufferWindowMemory (keeps last K turns)
# ======================================================
window_memory = ConversationBufferWindowMemory(
    k=1,  # only keep the last 1 interaction (user+AI)
    memory_key="chat_history", return_messages=True
)
window_chain = LLMChain(llm=llm, prompt=chat_template, memory=window_memory)

print("\n=== ConversationBufferWindowMemory (k=1) ===")
print(window_chain.run(input="Hello, who won the 2018 World Cup?"))
print(window_chain.run(input="Who was the captain of that team?"))
# Memory here will only include the LAST exchange.
# The first question/answer gets dropped after the second one.


# ======================================================
# 3. ConversationSummaryMemory (summarizes older context)
# ======================================================
summary_memory = ConversationSummaryMemory(
    llm=llm,  # uses LLM to summarize
    memory_key="chat_history", return_messages=True
)
summary_chain = LLMChain(llm=llm, prompt=chat_template, memory=summary_memory)

print("\n=== ConversationSummaryMemory ===")
print(summary_chain.run(input="Hello, who won the 2018 World Cup?"))
print(summary_chain.run(input="Can you remind me what you just told me?"))
# Instead of keeping everything verbatim, the LLM produces a summary.
# Memory will look like: "User asked about 2018 World Cup, AI said France won."


# ======================================================
# 4. ConversationKGMemory (facts in knowledge graph style)
# ======================================================
kg_memory = ConversationKGMemory(
    llm=llm,  # LLM helps extract entities/relations
    memory_key="chat_history"
)
kg_chain = LLMChain(llm=llm, prompt=chat_template, memory=kg_memory)

print("\n=== ConversationKGMemory ===")
print(kg_chain.run(input="My name is Alice. I live in Paris."))
print(kg_chain.run(input="Where do I live?"))
# Memory stores structured facts like:
#   ("Alice", "lives_in", "Paris")
# Useful when you want structured recall instead of raw text.


"""
? BufferMemory → keeps everything. Good for small/medium chats.
? BufferWindowMemory → keeps only last k turns. Lightweight, forgets old stuff.
? SummaryMemory → generates a summary of past conversation. Good for long-running chats.
? KGMemory → extracts facts and entities into a knowledge graph. Useful when structured recall is needed (like “Alice lives in Paris”).
"""