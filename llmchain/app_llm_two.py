from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
prompt_str = """Using the context provided, answer the user's question.
Context:
{context_a}
{context_b}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(prompt_str),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)
embedding = OpenAIEmbeddings()

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


retriever_a = vecstore_a.as_retriever()
retriever_b = vecstore_b.as_retriever()

retrieval = RunnableParallel(
    {
        "context_a": retriever_a,
        "context_b": retriever_b,
        "question": RunnablePassthrough(),
    }
)
output_parser = StrOutputParser()
chain = retrieval | prompt | llm | output_parser

result = chain.invoke(
    "what architecture does the model DeepSeek released in december use?"
)
result
