Great question 👍 — this is one of the most confusing new parts of LangChain 0.2+.

---

### 🔑 What `history_factory_config` does

* `RunnableWithMessageHistory` needs a way to **create or fetch** a chat history object (`BaseChatMessageHistory`) when you call `.invoke()` or `.stream()`.
* You tell it *how* to create that history object by passing a `get_session_history` function.
* But your `get_session_history` sometimes needs **extra parameters** besides just the `session_id`.

  * Example: your history object needs an `llm` or a database connection.

That’s where `history_factory_config` comes in:

* It’s a schema (list of `ConfigurableFieldSpec`) that defines what extra fields you can pass in `config` when you run `.invoke()`.
* Then, `RunnableWithMessageHistory` passes those into your `get_session_history` function.

---

### 🖼️ Example

#### Case 1: No `history_factory_config`

```python
pipeline_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history=lambda session_id: ConversationSummaryMessageHistory(llm),
    input_messages_key="query",
    history_messages_key="history",
)
```

* Here `get_session_history` only needs `session_id`.
* So when you call:

  ```python
  pipeline_with_history.invoke(
      {"query": "Hello"},
      config={"session_id": "abc123"}
  )
  ```

  it works fine — no `history_factory_config` needed.

---

#### Case 2: With `history_factory_config`

```python
def get_chat_history(session_id: str, llm: ChatOllama):
    return ConversationSummaryMessageHistory(llm=llm)

pipeline_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID for chat history",
        ),
        ConfigurableFieldSpec(
            id="llm",
            annotation=ChatOllama,
            name="LLM",
            description="The model to use for summarization",
            default=llm,
        )
    ]
)
```

* Now `get_chat_history` requires both `session_id` and `llm`.
* So when you call:

  ```python
  pipeline_with_history.invoke(
      {"query": "Hello"},
      config={"session_id": "abc123", "llm": llm}
  )
  ```

  it knows how to pass both args into `get_chat_history`.

---

### 🧠 Rule of Thumb: *When to use `history_factory_config`*

* **Don’t use it** if your history only needs `session_id` (most cases).
* **Use it** if your history needs extra objects (like an `llm`, a DB connection, or something configurable).
