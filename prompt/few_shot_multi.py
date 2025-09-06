from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_ollama import ChatOllama

# Connect to your local Ollama model
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# ---- Step 1: Few-shot examples ----
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{patient_input}"),
    ("ai", "{bot_response}")
])

examples = [
    {
        "patient_input": "I feel sick",
        "bot_response": "I’m sorry to hear that. Can you describe your main symptom?"
    },
    {
        "patient_input": "I have chest pain",
        "bot_response": "That sounds serious. Please share your age and medical history if available."
    },
    {
        "patient_input": "I feel weak and dizzy",
        "bot_response": "Thanks. Could you provide your temperature and heart rate if you know them?"
    },
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# ---- Step 2: Main triage prompt ----
main_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical triage assistant. "
               "Ask structured, step-by-step questions. "
               "Collect enough details (symptoms, vitals, history) before giving advice. "
               "Do not skip steps."),
    few_shot_prompt,
    ("human", "{user_query}")
])

# ---- Step 3: Simulate multi-turn flow ----
conversation_history = []  # keep track of messages

def ask_bot(user_input: str):
    """Send conversation (history + new user input) to bot"""
    conversation_history.append(("human", user_input))  # add latest patient message
    
    # Build prompt dynamically with history
    dynamic_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical triage assistant. Continue the conversation."),
        *conversation_history  # unfold past messages
    ])
    
    pipeline = dynamic_prompt | llm
    response = pipeline.invoke({"user_query": user_input})
    
    conversation_history.append(("ai", response.content))  # store bot reply
    return response.content

# ---- Demo ----
print("👨 Patient: I feel short of breath")
bot_reply = ask_bot("I feel short of breath")
print("🤖 Bot:", bot_reply)

print("\n👨 Patient: I'm 45, no medical history")
bot_reply = ask_bot("I'm 45, no medical history")
print("🤖 Bot:", bot_reply)

print("\n👨 Patient: My temperature is 38.5°C")
bot_reply = ask_bot("My temperature is 38.5°C")
print("🤖 Bot:", bot_reply)
