from datetime import datetime
import requests
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from IPython.display import display, Markdown


# Load local Ollama model (make sure the port is correct!)
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")


@tool
def get_location_from_ip() -> str:
    """Get location details (latitude, longitude, city, country) from IP address."""
    try:
        response = requests.get("https://ipinfo.io/json")
        info = response.json()
        if "loc" in info:
            latitude, longitude = info["loc"].split(",")
            result = (
                f"Latitude: {latitude},\n"
                f"Longitude: {longitude},\n"
                f"City: {info.get('city', 'N/A')},\n"
                f"Country: {info.get('country', 'N/A')}"
            )
            return result
        else:
            return "Error: could not determine location."
    except Exception as e:
        return f"Error occurred: {e}"


@tool
def get_data() -> str:
    """Return the current date and time as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Agent prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Register tools
tools = [get_data, get_location_from_ip]

# Create tool-calling agent
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example 1: math question
response = agent_executor.invoke({"input": "What is 5 multiplied by 12, then add 3?"})
print("Math response:", response["output"])

# Example 2: using tools
out = agent_executor.invoke(
    {
        "input": (
            "I have a few questions. "
            "What is the date and time right now? "
            "Also tell me my location details."
        )
    }
)
print("Tool response:", out["output"])
display(Markdown(out["output"]))
