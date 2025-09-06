from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

prompt_txt = "Answer this question: {query}"
prompt_template = ChatPromptTemplate.from_template(prompt_txt)

llmchain = prompt_template | llm

response = llmchain.invoke({"query": "Why do you work?"})

print("=== Model Output ===")
print(response.content)

print("\n=== Just formatting (no LLM call) ===")
print(prompt_template.format(query="What is AI?"))
print(prompt_template.format(query="Explain blockchain"))

