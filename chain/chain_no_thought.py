from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap

llm = ChatOllama(model='mistral', base_url='http://localhost:11434')

sys_prompt = """
Be a helpful AI assistant and answer the user's question.
"""

chat_template = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    ("user", "{query}")
])

pipeline = (
    RunnableMap({"query": lambda x: x["query"]}) 
    | chat_template 
    | llm 
    | StrOutputParser()
)

query = "How many keystrokes are needed to type the numbers from 1 to 500?"

response = pipeline.invoke({"query": query})

print(response)
