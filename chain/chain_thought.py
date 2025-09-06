from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model='mistral',base_url='http://localhost:11434')
sys_prompt = SystemMessagePromptTemplate.from_template(
    """
Be a helpful assistant and answer the user's questions.

You MUST answer the question directly without any 
extra text or explanations.
"""
)
chat_template_v = ChatPromptTemplate.from_messages([
    ("system", "Be a helpful assistant and answer the user's questions directly."),
    ("human", "{query}")
])

chat_template = ChatPromptTemplate.from_messages([
    sys_prompt,
    HumanMessagePromptTemplate.from_template("{query}")
])

# print(chat_template)

pipeline = chat_template | llm
query = "What is 2 + 2?"

response = pipeline.invoke({"query":query})

print(response.content)
query = "How many keystroke are needed to type the number from 1 to 500"

response = pipeline.invoke({"query":query})

print(response.content)
