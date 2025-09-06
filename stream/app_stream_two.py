import json
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


# === LLM setup ===
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")


# === Tools ===
@tool
def get_user_profile(user_id: str) -> dict:
    """Fetch user profile (id + name)."""
    # TODO: Replace with DB lookup
    return {"id": user_id, "name": "John Doe"}


@tool
def check_existing_consultation(user_id: str) -> dict:
    """Check if user has an active consultation."""
    # TODO: Replace with DB lookup
    return {"active": False, "doctorId": None, "doctorName": None}


@tool
def classify_triage(vitals: dict) -> dict:
    """Classify patient vitals into triage category."""
    hr = vitals.get("heart_rate", 0)
    temp = vitals.get("temperature", 0)
    sys = vitals.get("systolic_bp", 0)
    dia = vitals.get("diastolic_bp", 0)

    if hr > 120 or temp > 39 or sys > 180 or dia > 110:
        return {
            "triage": "Red",
            "reason": "Critical vitals detected.",
            "advice": "Seek immediate emergency medical attention.",
            "recommendedRole": "cardiologist",
        }
    return {
        "triage": "Green",
        "reason": "Vitals are within safe ranges.",
        "advice": "Routine follow-up only.",
        "recommendedRole": "general",
    }


@tool
def assign_doctor(user_id: str, role: str) -> dict:
    """Assign the least busy doctor for the given role."""
    # TODO: Replace with DB lookup
    return {"doctorId": "d456", "doctorName": "Dr. Alice"}


@tool
def final_answer(answer: str, result: dict) -> dict:
    """Provide the final structured answer to the user."""
    return {"answer": answer, "result": result}


tools = [get_user_profile, check_existing_consultation, classify_triage, assign_doctor, final_answer]


# === Prompt Template ===
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a structured health triage assistant. "
                "Follow this EXACT 5-step dialogue:\n"
                "1. Greet the user by fetching their name using get_user_profile.\n"
                "2. Wait for user response.\n"
                "3. Ask: 'I’ll quickly gather some details to check your health status. Ready to begin?'\n"
                "4. Wait for user response.\n"
                "5. Ask them to submit their vitals as JSON.\n"
                "6. When user provides vitals → classify using classify_triage.\n"
                "7. Check if they already have a consultation (check_existing_consultation).\n"
                "   - If yes, direct them to same doctor.\n"
                "   - If no, assign a doctor with assign_doctor.\n"
                "8. End with final_answer (include triage result + doctor info)."
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        {"user": "{input}"},
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# === Agent ===
agent: RunnableSerializable = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "agent_scratchpad": lambda x: x.get("agent_scratchpad", []),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)


# === Executor ===
class CustomAgentExecutor:
    def __init__(self, max_iterations: int = 5):
        self.chat_history: list[BaseMessage] = []
        self.max_iterations = max_iterations
        self.agent = agent
        self.name2tool = {t.name: t.func for t in tools}

    async def invoke(self, input: str, verbose=True) -> dict:
        agent_scratchpad = []
        count = 0

        while count < self.max_iterations:
            out = self.agent.invoke(
                {
                    "input": input,
                    "chat_history": self.chat_history,
                    "agent_scratchpad": agent_scratchpad,
                }
            )

            # If final answer, stop
            if out.tool_calls[0]["name"] == "final_answer":
                break

            # Execute tool
            tool_name = out.tool_calls[0]["name"]
            tool_args = out.tool_calls[0]["args"]
            tool_out = self.name2tool[tool_name](**tool_args)

            action_str = f"{tool_name} returned {tool_out}"
            if verbose:
                print(f"[Step {count}] {action_str}")

            agent_scratchpad.append(out)
            agent_scratchpad.append(
                {"role": "tool", "content": action_str, "tool_call_id": out.tool_calls[0]["id"]}
            )
            count += 1

        # Record final
        final_answer = out.tool_calls[0]["args"]
        final_answer_str = json.dumps(final_answer)
        self.chat_history.append({"input": input, "output": final_answer_str})
        self.chat_history.extend([HumanMessage(content=input), AIMessage(content=final_answer_str)])

        return final_answer


# === Run Demo ===
async def main():
    executor = CustomAgentExecutor()

    # Simulated conversation
    print("\n=== Conversation ===\n")

    # Step 1: Greet
    res1 = await executor.invoke("Hello", verbose=True)
    print("AI:", res1)

    # Step 2: User responds
    res2 = await executor.invoke("Hi!", verbose=True)
    print("AI:", res2)

    # Step 3: User submits vitals
    vitals = {
        "heart_rate": 120,
        "systolic_bp": 150,
        "diastolic_bp": 95,
        "temperature": 38.2,
        "weight": 75,
        "symptoms": "shortness of breath and chest pain",
    }
    res3 = await executor.invoke(json.dumps(vitals), verbose=True)
    print("AI:", res3)


if __name__ == "__main__":
    asyncio.run(main())
