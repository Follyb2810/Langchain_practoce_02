from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser


from langchain.prompts import PromptTemplate


# LLM
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# System prompt
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant called {name} that helps generate article titles.",
    input_variables=["name"]
)

# User prompt
user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a name for an article.
    The article is here for you to examine:
    -------
    {article}
    -------
    The name should be based on the context of the article.
    Be creative, but make sure the name is clear, catchy, and relevant.
    Only output the article name, no other explanation or text."""
    ,
    input_variables=["article"]
)

pt = PromptTemplate(
    template="You are {name}, an assistant.",
    input_variables=["name"]
)
SystemMessagePromptTemplate(prompt=pt)

# Combine prompts
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

# Build the chain
chain = (
    RunnableMap({
        "article": lambda x: x["article"],
        "name": lambda x: x["name"],
    })
    | chat_prompt
    | llm
    | StrOutputParser()  # ensures we just get the text string, not a ChatMessage object
)

# Run it
result = chain.invoke({"article": "The rise of AI in healthcare and its impact on doctors", "name": "Folly"})
print("Generated Title:", result)


