from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser  # corrected import

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

prompt_template = "Give me a small report on a {topic}"
prompt = PromptTemplate(input_variables=["topic"], template=prompt_template)

output_parser = StrOutputParser()

lcel = prompt | llm | output_parser

result = lcel.invoke({"topic": "retrieval augmented generation"})
print(result)

# --- If you want to still use LLMChain (legacy style) ---
# from langchain.chains import LLMChain
# chain = LLMChain(prompt=prompt, llm=llm, output_parser=output_parser)
# result = chain.invoke({"topic": "retrieval augmented generation"})
# print(result)
