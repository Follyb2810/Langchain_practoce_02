from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama

# Step 1: Define structured schema for triage result
class TriageResult(BaseModel):
    triage_level: str = Field(..., description="One of: green, yellow, amber, red")
    reasoning: str = Field(..., description="Short explanation of why this triage level was chosen")
    doctor_role: str = Field(..., description="Which doctor role should handle this case")

# Step 2: Create parser
parser = PydanticOutputParser(pydantic_object=TriageResult)

# Step 3: Prompt that enforces JSON output
prompt = PromptTemplate(
    template=(
        "You are a triage assistant. Classify the patient.\n\n"
        "Symptoms: {symptoms}\nVitals: {vitals}\n\n"
        "Return ONLY in JSON format:\n{format_instructions}"
    ),
    input_variables=["symptoms", "vitals"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Step 4: Call LLM
llm = Ollama(model="mistral")

_input = prompt.format(symptoms="chest pain, sweating", vitals="BP 90/60, HR 120")
output = llm.invoke(_input)

# Step 5: Parse result (safe)
try:
    parsed = parser.parse(output)
    print(parsed)
except Exception as e:
    print("Parsing failed, retry with stricter prompt:", e)



prompt = """
You are a triage bot. Output ONLY JSON with fields:
- triage_level: green, yellow, amber, red
- reasoning: why
- doctor_role: which doctor should handle
Patient input: chest pain, sweating, low blood pressure.
"""

print(llm.invoke(prompt))
