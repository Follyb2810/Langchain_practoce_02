from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from getpass import getpass
import os

print('start')
# Get key
openapi_key = getpass('please enter your key: ')
os.environ['OPENAI_API_KEY'] = openapi_key  # must be OPENAI_API_KEY

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")  # example
print(llm.invoke("Hello, how are you?"))
