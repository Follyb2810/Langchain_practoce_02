from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# import torch

# if torch.cuda.is_available():
#     print("GPU available:", torch.cuda.get_device_name(0))
# else:
#     print("No GPU detected. Using CPU.")

# invoke() = simple, template-driven call
# generate() = full chat messages, multi-turn capable

llm = ChatOllama(model='mistral', base_url='http://localhost:11434')

sys_prompt = "Act as a helpful assistant and give meaningful examples."
prompt = "Can you explain gen AI in 3 bullet points?"


messages = [
    [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=prompt)
    ]
]

response = llm.generate(messages)
#  response = llm.invoke(messages)

print(response.generations[0][0].text)
