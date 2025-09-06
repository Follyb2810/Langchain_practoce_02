from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model='mistral',base_url='http://localhost:11434')

prompt_text="Answer this : {question}"
prompt_template = ChatPromptTemplate.from_template(prompt_text)
format_message =prompt_template.format_messages(question='who are you')
response =llm.invoke(format_message)