from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# Example format: human asks → AI responds
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{patient_input}"),
    ("ai", "{bot_response}")
])

# Few-shot training examples for triage flow
examples = [
    {
        "patient_input": "I feel sick",
        "bot_response": "I’m sorry to hear that. Can you tell me your main symptoms?"
    },
    {
        "patient_input": "I have chest pain",
        "bot_response": "That sounds serious. Can you also tell me your age and medical history?"
    },
    {
        "patient_input": "I feel weak and dizzy",
        "bot_response": "Thanks for sharing. Could you provide your temperature and heart rate if available?"
    },
]

# Build the few-shot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# Now add your main system instruction + user message
main_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical triage assistant. Ask structured, step-by-step questions."),
    few_shot_prompt,  # Insert the few-shot examples here
    ("human", "{user_query}")
])

# Run pipeline
pipeline = main_prompt | llm

query = "I feel short of breath"
response = pipeline.invoke({"user_query": query})

print(response.content)
