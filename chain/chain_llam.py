from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser
llm = ChatOllama(model='mistral', base_url='http://localhost:11434')


#? structure data

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

chain = (
    RunnableMap({
        'article': lambda x: x['article'],
        'article_title': lambda x: x['article_title']
    })
    | final_message
    | llm
    | StrOutputParser()
    # | {"summary":lambda x:x['content']}
)

print(chain.invoke({"article": "This is a test article content", "article_title": "Test Title"}))
