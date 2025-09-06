from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache 

# Initialize LLM
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# Enable cache (repeated calls with same input won't re-hit the model)
set_llm_cache(InMemoryCache())

# -------------------------------
# One: PromptTemplate + format_messages
# Use this when you have a template with placeholders like {question}
# -------------------------------
prompt_text = "Answer this: {question}"
prompt_template = ChatPromptTemplate.from_template(prompt_text)

formatted_prompt = prompt_template.format_messages(question="What is 2+2?")
response = llm.invoke(formatted_prompt)
print("=== One (PromptTemplate) ===")
print(response.content, "\n")

# -------------------------------
# Two: Raw messages
# Use this when you want to directly control system/human roles
# Useful for chat history and instructions
# -------------------------------
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is 2+2?")
]
response = llm.invoke(messages)
print("=== Two (Raw Messages) ===")
print(response.content, "\n")

# -------------------------------
# Three: Chain (PromptTemplate | LLM)
# Use this when you want to build a reusable pipeline
# (like in production apps where you call the same chain many times)
# -------------------------------
llmchain = prompt_template | llm
response = llmchain.invoke({"question": "Why do you work?"})
print("=== Three (Chaining) ===")
print(response.content, "\n")

# -------------------------------
# Four: Generate (batch calls / multiple inputs)
# Use this when you want to run multiple queries at once
# or need token usage and raw metadata
# -------------------------------
batch_inputs = [
    [HumanMessage(content="What is 2+2?")],
    [HumanMessage(content="Why is the sky blue?")]
]

result = llm.generate(batch_inputs)

print("=== Four (Generate) ===")
for i, gen in enumerate(result.generations):
    print(f"Input {i+1}: {batch_inputs[i][0].content}")
    print("Output:", gen[0].text, "\n")

# If you want usage info (tokens), `result.llm_output` may contain it
print("Raw LLMResult:", result.llm_output)
