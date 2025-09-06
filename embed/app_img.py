from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from skimage import io
import matplotlib.pyplot as plt
from langchain_core.runnables import RunnableLambda
from langchain.schema.runnable import RunnableMap
from langchain_ollama import ChatOllama

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

# Image prompt template (fix: single string, not tuple)
image_prompt = PromptTemplate(
    input_variables=['article'],
    template="Generate a prompt (max 500 chars) for an image based on the following article: {article}"
)

# Function to fetch + display image
def generate_image(image_prompt: str):
    image_url = DallEAPIWrapper().run(image_prompt)
    image_data = io.imread(image_url)
    plt.imshow(image_data)
    plt.axis('off')
    plt.show()
    return image_url  # return in case you want to log/store it

image_gen_runnable = RunnableLambda(generate_image)

# Chain
chain = (
    RunnableMap({
        "article": lambda x: x["article"],
        "article_title": lambda x: x["article_title"]
    })
    | image_prompt          # format article into image prompt string
    | llm                   # generate refined text
    | StrOutputParser()     # extract plain string from AIMessage
    | image_gen_runnable    # pass to image generator
)

print(
    chain.invoke({"article": "This is a test article content", "article_title": "Test Title"})
)
