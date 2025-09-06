# --------------------------------------------------------------
# STEP 0: Imports
# --------------------------------------------------------------
# Embeddings model used to turn text into vectors. NOTE: This specific class
# uses OpenAI and expects OPENAI_API_KEY in your environment.
from langchain.embeddings import OpenAIEmbeddings

# An in-memory vector store (keeps vectors in RAM). Great for demos and tests.
# (In newer LangChain versions this may live under langchain_community.vectorstores)
from langchain.vectorstores import DocArrayInMemorySearch

# Prompt building utilities for chat-style prompts:
# - ChatPromptTemplate: the overall template
# - SystemMessagePromptTemplate: system instruction block (sets behavior)
# - HumanMessagePromptTemplate: the user's question/message template
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Runnable building blocks:
# - RunnablePassthrough: forwards the original input unchanged
# - RunnableParallel: runs multiple runnables in parallel and returns a dict of their outputs
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# LLM wrapper for local Ollama server (here we target the "mistral" model)
from langchain_ollama import ChatOllama

# Converts the LLM's structured result to a plain string.
# NOTE: In LangChain >= 0.2, prefer: `from langchain_core.output_parsers import StrOutputParser`
from langchain.schema.output_parser import StrOutputParser


# --------------------------------------------------------------
# STEP 1: Initialize the LLM
# --------------------------------------------------------------
# We configure an Ollama chat model that will answer our prompt.
# `base_url` points at your local Ollama instance (default localhost:11434).
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")


# --------------------------------------------------------------
# STEP 2: Build the prompt template
# --------------------------------------------------------------
# This string defines how we want to arrange the retrieved context alongside the question.
# We expose two context slots (context_a and context_b) to demonstrate merging results
# from multiple retrievers/vector stores.
prompt_str = """Using the context provided, answer the user's question.
Context:
{context_a}
{context_b}
"""

# We create a chat-style prompt with two messages:
# 1) System message: holds our instructions and places for the retrieved context
# 2) Human message: the user's actual question
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(prompt_str),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)


# --------------------------------------------------------------
# STEP 3: Embeddings + Vector stores
# --------------------------------------------------------------
# The embeddings model transforms raw text into vectors so the vector store can do similarity search.
embedding = OpenAIEmbeddings()

# Create two separate in-memory vector stores with small toy corpora.
# Each `from_texts` call indexes the texts so they can be retrieved later by semantic similarity.
vecstore_a = DocArrayInMemorySearch.from_texts(
    ["half the info is here", "DeepSeek-V3 was released in December 2024"],
    embedding=embedding,
)
vecstore_b = DocArrayInMemorySearch.from_texts(
    [
        "the other half of the info is here",
        "the DeepSeek-V3 LLM is a mixture of experts model with 671B parameters",
    ],
    embedding=embedding,
)


# --------------------------------------------------------------
# STEP 4: Turn vector stores into retrievers
# --------------------------------------------------------------
# A retriever is a simple interface: you give it a query string -> it returns relevant Documents.
retriever_a = vecstore_a.as_retriever()
retriever_b = vecstore_b.as_retriever()


# --------------------------------------------------------------
# STEP 5: Wire up retrieval with RunnableParallel
# --------------------------------------------------------------
# RunnableParallel creates a dict-shaped output by running each value (a Runnable) in parallel.
# Keys on the left ("context_a", "context_b", "question") become keys in the output dict.
# - For "context_a": we run retriever_a on the *same input* we pass into the chain
# - For "context_b": same idea with retriever_b
# - For "question": we just forward the original input unchanged using RunnablePassthrough()
#
# IMPORTANT PRACTICAL NOTE:
# retrievers return a list of Document objects. When plugging into a string prompt slot
# like {context_a}, you typically want to *format* those docs to a string, e.g. join their
# page_content with newlines. For simplicity, this example relies on Python's default str()
# of the list/Document, but in real code you'd add a small formatter Runnable such as:
#
#   from langchain_core.runnables import RunnableLambda
#   def format_docs(docs): return "\n\n".join(d.page_content for d in docs)
#   retrieval = RunnableParallel({
#       "context_a": retriever_a | RunnableLambda(format_docs),
#       "context_b": retriever_b | RunnableLambda(format_docs),
#       "question": RunnablePassthrough(),
#   })
#
retrieval = RunnableParallel(
    {
        "context_a": retriever_a,
        "context_b": retriever_b,
        "question": RunnablePassthrough(),
    }
)


# --------------------------------------------------------------
# STEP 6: Output parser
# --------------------------------------------------------------
# Ensures we end up with a clean string result from the LLM (instead of a Message object).
output_parser = StrOutputParser()


# --------------------------------------------------------------
# STEP 7: Compose the full LCEL pipeline
# --------------------------------------------------------------
# LCEL (`|`) chains steps left-to-right:
# 1) `retrieval`: given the user's question, fetch context_a and context_b and pass the question through
# 2) `prompt`: fills the {context_a}, {context_b}, {question} placeholders from the dict produced by retrieval
# 3) `llm`: sends the formatted prompt to the Ollama LLM
# 4) `output_parser`: converts the LLM output to a plain string
chain = retrieval | prompt | llm | output_parser


# --------------------------------------------------------------
# STEP 8 (Optional): How to call the chain
# --------------------------------------------------------------
# The chain expects the *user's question* as its input (a single string), because
# RunnablePassthrough and the retrievers both consume the same top-level input.
#
# Example:
# answer = chain.invoke("What is DeepSeek-V3 and when was it released?")
# print(answer)
#
# In production, remember to:
# - Provide OPENAI_API_KEY for embeddings
# - Consider formatting retrieved docs before inserting into the prompt
# - Swap DocArrayInMemorySearch for a persistent/vector DB in real apps
result = chain.invoke(
    "what architecture does the model DeepSeek released in december use?"
)
result