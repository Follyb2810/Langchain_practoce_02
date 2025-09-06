from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

system_prompt = """
Answer the user query's base on the context below.
If you cannot answer the question using the provided 
information answer with "i don't know".

Context: {context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{query}")
])

pipeline = (
    # RunnableMap(
        {
        "query": lambda x: x["query"],
        "context": lambda x: x["context"].strip()
    }
    # )
    | prompt_template
    | llm
)

context = """
Follyb Ai is a Ai company developing tooling for engineers.
Their focus is on language AI with the team having strong
expertise in building AI Agents and a strong background in information retrieval.

The company is behind several open source frameworks, most notably Semantic Router
and Semantic Chunker.
They also have an AI platform providing engineers with tooling to help them build with AI.
Follyb Ai became LangChain experts in August 2025 after a long track of learning,
implementing, and building solutions in AI.
"""
query = "What does Follyb Ai do?"

response = pipeline.invoke({"query": query, "context": context}).content
print(response.content)
