from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# -----------------------------
# Step 1: Connect to Mistral via Ollama Docker
# Make sure your Docker container is running and port matches
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# -----------------------------
# Part A: Using invoke() — template-driven
print("\n=== Using invoke() ===")

# Create a simple prompt template with a placeholder
prompt_template = ChatPromptTemplate.from_template("Answer this: {question}")

# Pipe template to LLM
chain = prompt_template | llm

# Call invoke with variable dict
response_invoke = chain.invoke({"question": "What is AI?"})

# Output
print("invoke() result:", response_invoke.content)

# -----------------------------
# Part B: Using generate() — chat messages
print("\n=== Using generate() ===")

# Build a system + human message sequence
messages = [
    [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain AI in 3 bullet points")
    ]
]

# Call generate() with a list of message sequences
response_generate = llm.generate(messages)

# Output the generated text
print("generate() result:", response_generate.generations[0][0].text)

# -----------------------------
# Extra: multi-turn chat example
print("\n=== Multi-turn generate() ===")

multi_turn_messages = [
    [
        SystemMessage(content="You are a medical triage assistant."),
        HumanMessage(content="Hello, my heart rate is 120")
    ]
]

multi_response = llm.generate(multi_turn_messages)
print("multi-turn output:", multi_response.generations[0][0].text)
