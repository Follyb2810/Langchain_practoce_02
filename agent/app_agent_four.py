from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory

# === 1. Define the LLM ===
# ChatOllama connects to a local Ollama model (here "mistral").
# base_url points to your local Ollama server.
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")


# === 2. Define Memory ===
# Memory lets the LLM "remember" previous conversation turns.
# Here we use ConversationBufferMemory:
# - Stores the ENTIRE conversation history as a growing buffer (like a chat log).
# - `memory_key='chat_history'` ensures the memory gets injected wherever
#   the prompt has MessagesPlaceholder(variable_name="chat_history").
# - `return_messages=True` makes it return structured HumanMessage/AIMessage objects,
#   instead of just a long string (better for ChatPromptTemplate).
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# === 3. Define Prompt Template ===
# A chat template defines the "shape" of each LLM call.
# It mixes fixed messages with placeholders for memory and runtime values.
chat_template = ChatPromptTemplate.from_messages(
    [
        # System role → used for fixed instructions to guide behavior.
        ("system", "you are a helpful assistant"),

        # MessagesPlaceholder → this is where the conversation history
        # (stored in memory.chat_history) gets injected each time.
        # Without this, the model would "forget" prior context.
        MessagesPlaceholder(variable_name="chat_history"),

        # Human input placeholder → the latest user query will go here.
        ("human", "{input}"),

        # Scratchpad placeholder → used when building an Agent.
        # During tool use, the agent writes intermediate reasoning steps here.
        # Not needed for a simple chatbot, but useful for Agents.
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# === 4. Different Memory Types (explained) ===
# - ConversationBufferMemory:
#   Keeps the FULL history. Good for small/medium conversations.
#   But can get too long → may hit token limits.
#
# - ConversationSummaryMemory:
#   Summarizes older parts of the conversation to save context.
#   Useful for long-running chats where you can’t keep everything verbatim.
#
# - ConversationBufferWindowMemory:
#   Keeps ONLY the last "k" interactions (like a sliding window).
#   Good when you only care about recent context (e.g., chatbots).
#
# - ConversationKGMemory:
#   Stores facts as a knowledge graph (triplets). Helps with structured recall.
#
# Each of these can be swapped in, as long as you set
# memory_key="chat_history" to match your MessagesPlaceholder.


# === 5. Example Usage (not full agent, just chat) ===
# User message → memory saves it → template inserts it → LLM answers.
# You'd normally wire this into a chain/agent executor, e.g.:
# chain = LLMChain(llm=llm, prompt=chat_template, memory=memory)
# chain.run(input="Hello, who won the 2018 World Cup?")
