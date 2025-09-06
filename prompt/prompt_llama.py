from langchain.prompts import HumanMessagePromptTemplate,SystemMessagePromptTemplate,ChatPromptTemplate
# how ai should act
system_prompt =SystemMessagePromptTemplate.from_template('You are an Ai assistence that help generate article title.')
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
print(first_prompt.format(article='Test String'))