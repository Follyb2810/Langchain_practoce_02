from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Local model via Ollama
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# Define state
class TriageState(TypedDict):
    user_input: str
    form_data: dict
    result: dict

# Nodes
def greet(state: TriageState):
    return {"user_input": f"Hello {state['user_input']}! How can I assist you today?"}

def intake(state: TriageState):
    return {"user_input": "I’ll quickly gather some details to check your health status. Ready to begin?"}

def collect_form(state: TriageState):
    # simulate form (you’ll replace this with real frontend form submission)
    return {
        "form_data": {
            "heart_rate": 120,
            "systolic_bp": 150,
            "diastolic_bp": 95,
            "temperature": 38.2,
            "weight": 75,
            "symptoms": "shortness of breath and chest pain"
        }
    }

def classify(state: TriageState):
    vitals = state["form_data"]
    # Very basic triage rule (you’ll expand this)
    if vitals["heart_rate"] > 110 or vitals["systolic_bp"] > 140 or "chest pain" in vitals["symptoms"]:
        triage = "Red"
        reason = "The patient's vitals indicate critical abnormality."
        advice = "Seek immediate emergency medical attention."
        role = "cardiologist"
    else:
        triage = "Green"
        reason = "Vitals are stable."
        advice = "No urgent care needed."
        role = "general"

    return {
        "result": {
            "triage": triage,
            "reason": reason,
            "advice": advice,
            "recommendedRole": role
        }
    }

# Build graph
graph = StateGraph(TriageState)
graph.add_node("greet", greet)
graph.add_node("intake", intake)
graph.add_node("form", collect_form)
graph.add_node("classify", classify)

graph.set_entry_point("greet")
graph.add_edge("greet", "intake")
graph.add_edge("intake", "form")
graph.add_edge("form", "classify")
graph.add_edge("classify", END)

# Compile
app = graph.compile()

# Run
result = app.invoke({"user_input": "John"})
print(result)
