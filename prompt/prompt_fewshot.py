from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableMap 
from langchain_ollama import ChatOllama
from IPython.display import display,Markdown

# ------------------------------
# Initialize the Ollama LLM
# ------------------------------
llm = ChatOllama(model='mistral', base_url='http://localhost:11434')

# ------------------------------
# Base system prompt template
# ------------------------------
system_prompt = """
Answer the user query based on the context below.
If you cannot answer the question using the provided 
information answer with "I don't know".

Context: {context}
"""

# Prompt template with system + user messages
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{query}")
])

# ------------------------------
# Few-shot examples setup
# ------------------------------

# Example prompt format for few-shot learning
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

examples = [
    {"input": "who are you", "output": "I am an AI."},
    {"input": "what is your job?", "output": "I help answer questions."},
    {"input": "where are you from?", "output": "I was created by developers."},
]


few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

print("=== Few Shot Prompt Example ===")
print(few_shot_prompt.format())

# ------------------------------
# Updated system prompt
# ------------------------------
new_sys_prompt = """
Answer the query based on the context below.
If you cannot answer the question using the provided 
information answer with "I don't know".

Always answer in markdown. When doing so please
provide headers, short summaries, follow with bullet
points, then conclude.

Context: {context}
"""

# Update the system message of the prompt_template
prompt_template.messages[0].prompt.template = new_sys_prompt

# ------------------------------
# Pipeline setup
# ------------------------------
pipeline = (
    RunnableMap( 
        {
            "query": lambda x: x["query"],
            "context": lambda x: x["context"].strip()
        }
    )
    | prompt_template  
    | llm              
)

# ------------------------------
# Example usage
# ------------------------------

# Define some context for testing
context = "Follyb AI is a company that builds telemedicine and AI-powered healthcare chatbots."

query = "What does Follyb AI do?"

# Run pipeline
out = pipeline.invoke({'query': query, 'context': context})

# Output final result
print("\n=== LLM Output ===")
print(out.content)
display(Markdown(out.content))
