from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate


prompt = """
Answer the user query's base on the context below.
If you cannot answer the question using the provided 
information answer with "i don't know". 

Context: {context}
"""


prompt_template = ChatPromptTemplate.from_messages([
    {"role": "system", "content": prompt},   
    {"role": "user", "content": "{query}"}   
])

print(prompt_template.input_variables)
print(prompt_template.messages)
