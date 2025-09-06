from langchain.prompts import HumanMessagePromptTemplate,SystemMessagePromptTemplate,ChatPromptTemplate
from langchain_ollama import ChatOllama
# how ai should act

llm = ChatOllama(model='mistral',base_url='http://localhost:11434')

system_prompt =SystemMessagePromptTemplate.from_template(
    "You are an Ai assistence called {name} that help generate article title.",
    input_variable=['name']
    )
#user input
user_prompt = HumanMessagePromptTemplate.from_template(
    '''you are tasked with creating a name for article.
    The article is here for you to examine :
    -------

    {article} 
     
    --------
    The name should be base on the context of the article.
    be creative, but make sure the name are clear,catch and relevant to the article.
    Only output the aricle name, no other explanation or text to be provided
''',
input_variable=['article']
)

user_prompt.format(article='Test String').content
print(user_prompt)
print(user_prompt.format(article='Test String').content)

first_prompt = ChatPromptTemplate.from_messages([system_prompt,user_prompt])
print(first_prompt.format(name="follyb",article='Test String'))

chain = (
    {
        "article":lambda x:x['article'],
        "name":lambda x:x['name'],
    }
    | first_prompt 
    | llm 
    | {"article_tirle":lambda x :x.content}
)

article_title_msg = chain.invoke({'article':'what is the day in all',"name":'folly'})
print(article_title_msg)

