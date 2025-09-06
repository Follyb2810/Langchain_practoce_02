from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema.runnable import RunnableMap
from pydantic import BaseModel, Field
from langchain.schema.output_parser import StrOutputParser

llm = ChatOllama(model='mistral', base_url='http://localhost:11434')

# Prompts
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant that builds good articles."
)

user_prompt = HumanMessagePromptTemplate.from_template(
    """
You are tasked with creating description for articles.
The article is here for you to examine :

--------
{article}

_________

Here is the article title '{article_title}'.

Output the SEO-friendly article description.
Do not output anything more than the description.
Make sure we dont exceed 120 characters.
""",
    input_variables=['article','article_title']
)

final_message = ChatPromptTemplate.from_messages([system_prompt, user_prompt])


# Pydantic model for structured output
class Paragraph(BaseModel):
    original_paragraph: str = Field(description='the original paragraph')
    edited_paragraph: str = Field(description='the improved edited paragraph')
    feedback: str = Field(description='constructive feedback on the original paragraph')
    rating: int = Field(description='rate the edited paragraph')


# LLM with structured output
structure_llm = llm.with_structured_output(Paragraph)


# Runnable chain
chain = (
    RunnableMap({
        "article": lambda x: x["article"],
        "article_title": lambda x: x["article_title"]
    })
    | final_message
    | structure_llm

)

chain_string = (
    RunnableMap({
        "article": lambda x: x["article"],
        "article_title": lambda x: x["article_title"]
    })
    | final_message
    | structure_llm
    | (lambda p: p.dict())  
    | StrOutputParser()      
)


# Run it
print(
    chain.invoke({"article": "This is a test article content", "article_title": "Test Title"})
)
