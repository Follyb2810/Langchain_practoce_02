# Load environment variables from a .env file
# .env is a hidden file where you keep secrets like API keys, database URLs, etc.
from dotenv import load_dotenv
import os

# Import LangChain Ollama connector (to call local LLMs like Mistral)
from langchain_ollama import ChatOllama

# Import LangSmith traceable decorator (to track functions in LangSmith dashboard if enabled)
from langsmith import traceable

# Standard Python libraries
import random      # to generate random numbers
import time        # to pause execution (simulate work)
from tqdm.auto import tqdm  # to show a progress bar in the loop


# 1. Load variables from the .env file into environment variables
load_dotenv()

# 2. Read the environment variables
LANGSMITH = os.getenv('LANGSMITH')                 # maybe an API key
LANGSMITH_ENDPOINT = os.getenv('LANGSMITH_ENDPOINT')  # the service URL
LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT')    # project name

# 3. Print them out (to check they loaded correctly)
print(LANGSMITH)
print(LANGSMITH_ENDPOINT)
print(LANGSMITH_PROJECT)


# 4. Initialize a local LLM (Mistral model running in Ollama)
#    base_url points to where Ollama server is running (default localhost:11434)
llm = ChatOllama(model='mistral', base_url='http://localhost:11434')

# 5. Call the LLM and print the response
print(llm.invoke('Why cant you access my system'))


# --- Functions (all decorated with @traceable so LangSmith can record their runs) ---

@traceable
def generate_random_number():
    """Return a random integer between 0 and 100."""
    return random.randint(0, 100)

@traceable
def generate_random_string(input: str):
    """
    Append a random number (0–5) to the given string.
    Sleep for `number` seconds to simulate delay.
    """
    number = random.randint(0, 5)
    time.sleep(number)  # simulate a task that takes `number` seconds
    return f"{input}-{number}"

@traceable
def generate_rand_err():
    """
    Randomly raise an error (50% chance).
    If number == 0 → raise ValueError
    If number == 1 → return 'No error'
    """
    number = random.randint(0, 1)
    if number == 0:
        raise ValueError('Random Error')
    else:
        return 'No error'
    

# --- Run loop with progress bar ---
# tqdm(range(10)) shows a progress bar from 0 to 9
for i in tqdm(range(10)):
    # Call function that generates a number
    generate_random_number()
    
    # Call function that generates a string with delay
    generate_random_string('follyb')
    
    # Try function that sometimes raises error
    try:
        generate_rand_err()
    except ValueError:
        # Ignore the error so the loop continues
        pass
