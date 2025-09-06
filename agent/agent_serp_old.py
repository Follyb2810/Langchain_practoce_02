from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI

# Load LLM
llm = OpenAI(temperature=0)

# Load SerpAPI tool
tools = load_tools(["serpapi"])

# Old style: quick agent
agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True
)

# Run query
agent.run("Who is the president of France?")
