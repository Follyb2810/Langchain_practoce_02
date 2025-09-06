from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Step 1: Define the template
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),   # dynamic chat history
        ("human", "{input}"),                                # latest user input
        ("placeholder", "{agent_scratchpad}")                # agent scratchpad
    ]
)

# Step 2: Simulated runtime values
chat_history = [
    HumanMessage(content="Hello, who won the world cup in 2018?"),
    AIMessage(content="France won the 2018 FIFA World Cup."),
]

input_text = "Who was the captain of that team?"
agent_scratchpad = "Thinking... The answer should be Hugo Lloris, need to verify."

# Step 3: Format the template with real values
formatted_messages = chat_template.format_messages(
    chat_history=chat_history,
    input=input_text,
    agent_scratchpad=agent_scratchpad
)

# Step 4: See the final prompt
for msg in formatted_messages:
    print(f"{msg.type.upper()}: {msg.content}")
