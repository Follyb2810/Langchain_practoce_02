from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser

llm = ChatOllama(model='mistral',base_url='http://localhost:11434')

sys_prompt = """
Be a helpful assistant and answer the user question.

To answer the question, you must:

- List systematically and in precise details all
  sub problems that need to be solved to answer the question.
- Solve each sub problem INDIVIDUALLY and in sequence.
- Finally, use everything you have worked through to provide
  the final answer.

Context: {context}
"""

chat_template = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    ("human", "{query}")
])

pipeline = chat_template | llm 
# | StrOutputParser()

query = "How many keystrokes are needed to type the numbers from 1 to 500?"
context = "Typing numbers on a standard keyboard."

response = pipeline.invoke({"query": query, "context": context})

print("\n=== Response ===")
print(response.content)
