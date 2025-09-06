from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


system_prompt = SystemMessagePromptTemplate.from_template(
    """
You are a helpful AI assistant.
Always answer based only on the given context.
If the answer cannot be found in the context, reply strictly with "i don't know".
"""
)

user_prompt = HumanMessagePromptTemplate.from_template(
    """
Context:
--------
{context}
--------

User Question: {query}
"""
)

prompt_template = ChatPromptTemplate.from_messages([
    system_prompt,   
    user_prompt      
])


print(prompt_template.input_variables)


llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

chain = prompt_template | llm

result = chain.invoke({
    "context": "Python has try/except for handling errors.",
    "query": "Does Python support try catch?"
})

print(result.content)
