from datetime import datetime
import requests
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_ollama import ChatOllama
from IPython.display import display, Markdown
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Load local Ollama model
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# ⚠️ Replace with your actual OpenWeather API key


@tool
def get_location_from_ip() -> str:
    """Get location details (latitude, longitude, city, country) from IP address."""
    try:
        response = requests.get("https://ipinfo.io/json")
        info = response.json()
        if "loc" in info:
            latitude, longitude = info["loc"].split(",")
            result = {
                "latitude": latitude,
                "longitude": longitude,
                "city": info.get("city", "N/A"),
                "country": info.get("country", "N/A"),
            }
            return str(result)  # return as string for LLM
        else:
            return "Error: could not determine location."
    except Exception as e:
        return f"Error occurred: {e}"


@tool
def get_weather() -> str:
    """Fetch current weather in Celsius for the user's IP-based location."""
    try:
        # Step 1: get location first
        response = requests.get("https://ipinfo.io/json")
        info = response.json()
        if "loc" not in info:
            return "Error: could not determine location."

        latitude, longitude = info["loc"].split(",")

        # Step 2: call OpenWeather API
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"lat={latitude}&lon={longitude}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        weather_res = requests.get(url).json()

        if "main" in weather_res:
            temp = weather_res["main"]["temp"]
            desc = weather_res["weather"][0]["description"]
            city = weather_res.get("name", "Unknown City")
            return f"The current weather in {city} is {temp}°C with {desc}."
        else:
            return f"Error fetching weather: {weather_res}"

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
tools = [get_data, get_location_from_ip, get_weather]

# Create tool-calling agent
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Example 1: math question
response = agent_executor.invoke({"input": "What is 5 multiplied by 12, then add 3?"})
print("Math response:", response["output"])

# Example 2: tools in action (date + weather)
out = agent_executor.invoke(
    {
        "input": (
            "I have a few questions. "
            "What is the date and time right now? "
            "Also, tell me the weather where I am."
        )
    }
)
print("Tool response:", out["output"])
display(Markdown(out["output"]))
