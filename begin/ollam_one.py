from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from getpass import getpass
import os

openapi_key = getpass('please enter you key')
os.environ['openapi_key'] = openapi_key

# Local LLM
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# State definition
class TriageState(TypedDict):
    patient_name: str
    user_input: str
    last_consultation: dict
    vitals: dict
    result: dict

# --- Nodes ---
def greet_node(state: TriageState):
    return {"user_input": f"Hello {state['patient_name']}! How can I assist you today?"}

def fetch_patient_node(state: TriageState):
    # Fake DB call (replace with Mongo/SQL)
    db = {
        "John": {"date": "2025-08-01", "triage": "Yellow", "reason": "High BP"},
        "Mary": None
    }
    last = db.get(state["patient_name"], None)
    return {"last_consultation": last}

def followup_node(state: TriageState):
    if state["last_consultation"]:
        return {"user_input": f"I see your last visit on {state['last_consultation']['date']} "
                              f"was {state['last_consultation']['triage']} due to {state['last_consultation']['reason']}. "
                              f"Are you still experiencing similar issues?"}
    return {"user_input": "Let’s gather some details to check your health. Can you provide vitals?"}

def vitals_node(state: TriageState):
    # Example: assume frontend sends vitals in state
    return {"vitals": {
        "heart_rate": 120,
        "systolic_bp": 150,
        "diastolic_bp": 95,
        "temperature": 38.2,
        "weight": 75,
        "symptoms": "shortness of breath and chest pain"
    }}

def triage_node(state: TriageState):
    # Use guideline prompt (could be enhanced with RAG)
    prompt = ChatPromptTemplate.from_template("""
    You are a medical triage assistant. Classify the patient based on these vitals:
    {vitals}

    Return JSON:
    {{
      "triage": "...",
      "reason": "...",
      "advice": "...",
      "recommendedRole": "..."
    }}
    """)
    response = llm.invoke(prompt.format_messages(vitals=state["vitals"]))
    return {"result": response.content}

# --- Build Graph ---
graph = StateGraph(TriageState)

graph.add_node("greet", greet_node)
graph.add_node("fetch_patient", fetch_patient_node)
graph.add_node("followup", followup_node)
graph.add_node("vitals", vitals_node)
graph.add_node("triage", triage_node)

graph.set_entry_point("greet")
graph.add_edge("greet", "fetch_patient")
graph.add_edge("fetch_patient", "followup")
graph.add_edge("followup", "vitals")
graph.add_edge("vitals", "triage")
graph.add_edge("triage", END)

app = graph.compile()

# --- Run Example ---
result = app.invoke({"patient_name": "John"})
print(result)
