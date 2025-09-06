```ts
import { ChatOllama } from "langchain/llms/ollama";
import { ChatPromptTemplate, MessagesPlaceholder } from "langchain/prompts";
import { ConversationBufferMemory } from "langchain/memory";
import { tool, createToolCallingAgent, AgentExecutor } from "langchain/agents";

// === 1. Define tools ===
const add = tool(async ({ x, y }: { x: number; y: number }) => x + y, {
  name: "add",
  description: "Add two numbers",
});

const multiply = tool(async ({ x, y }: { x: number; y: number }) => x * y, {
  name: "multiply",
  description: "Multiply two numbers",
});

const exponentiate = tool(
  async ({ x, y }: { x: number; y: number }) => Math.pow(x, y),
  {
    name: "exponentiate",
    description: "Raise x to the power of y",
  }
);

const subtract = tool(
  async ({ x, y }: { x: number; y: number }) => y - x,
  {
    name: "subtract",
    description: "Subtract x from y",
  }
);

// === 2. Define LLM ===
const llm = new ChatOllama({
  model: "mistral",
  baseUrl: "http://localhost:11434",
});

// === 3. Define Memory ===
const memory = new ConversationBufferMemory({
  memoryKey: "chat_history",
  returnMessages: true,
});

// === 4. Prompt Template ===
const chatTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant"],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
  ["placeholder", "{agent_scratchpad}"],
]);

// === 5. Create agent and executor ===
const tools = [add, multiply, exponentiate, subtract];
const agent = createToolCallingAgent({ llm, tools, prompt: chatTemplate });

const agentExecutor = new AgentExecutor({
  agent,
  tools,
  memory,
  verbose: true,
});

// === 6. Run the agent ===
const response = await agentExecutor.invoke({
  input: "what is 10.07 multiplied by 7.687",
});

console.log("Agent Response:", response);

```