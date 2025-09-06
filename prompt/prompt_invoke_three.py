from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

prompt = """
Answer the user query's base on the context below.
If you cannot answer the question using the provided 
information answer with "i don't know". 

Context: {context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt),
    HumanMessagePromptTemplate.from_template("{query}")
])

pipeline = prompt_template | llm

context = """
Follyb Ai is a Ai company developing tooling for engineers...
"""
query = "What does Follyb Ai do?"

response = pipeline.invoke({"query": query, "context": context})
print(response.content)
