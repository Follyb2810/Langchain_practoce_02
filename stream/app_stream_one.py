from langchain_ollama import ChatOllama
import asyncio

llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

result = llm.invoke("what is my name? keep it short")
print("Sync result:", result.content)


async def main():
    tokens = []
    async for chunk in llm.astream("what is nlp? keep it short and precise"):
        tokens.append(chunk.content)
        print(chunk.content, end="|", flush=True)

    print("\n\nFinal assembled:", "".join(tokens))


asyncio.run(main())
