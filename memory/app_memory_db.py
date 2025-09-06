"""
postgres_chat_history_example.py

Requirements:
    pip install sqlalchemy psycopg2-binary langchain langchain-ollama
(Adjust langchain imports to your installed version; this code uses the Runnable-style APIs.)
"""

import datetime
from typing import List, Dict, Any

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Table,
    select,
    insert,
    update,
)
from sqlalchemy.orm import declarative_base, Session, relationship, sessionmaker

# LangChain imports (adjust if your environment differs)
from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# NOTE: RunnableWithMessageHistory expects a "get_session_history" callable that returns
# an object with .messages, add_user_message, add_ai_message, clear (typical shape).


Base = declarative_base()

# -----------------------
# DB models
# -----------------------
class SessionMeta(Base):
    __tablename__ = "session_meta"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(128), unique=True, nullable=False, index=True)
    last_visit = Column(DateTime, nullable=True)
    doctor = Column(String(256), nullable=True)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(128), index=True, nullable=False)
    role = Column(String(16), nullable=False)  # "user" or "ai"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


# -----------------------
# Postgres-backed chat history
# -----------------------
class PostgresChatMessageHistory:
    """
    Minimal chat-history object to satisfy RunnableWithMessageHistory expectations:
      - .messages -> list of message-like objects (with .type and .content)
      - add_user_message(str)
      - add_ai_message(str)
      - clear()
    """

    def __init__(self, db_session_factory, session_id: str):
        self._db_session_factory = db_session_factory
        self.session_id = session_id

    # Message container objects compatible with langchain message shape:
    # We'll create simple objects with .type ('human'|'ai') and .content attributes
    class _Msg:
        def __init__(self, type_, content, created_at=None):
            self.type = type_  # 'human' or 'ai'
            self.content = content
            self.created_at = created_at

        def to_dict(self):
            return {"type": self.type, "content": self.content, "created_at": self.created_at}

    @property
    def messages(self) -> List[_Msg]:
        """Return all messages for this session ordered by created_at."""
        with self._db_session_factory() as db:
            rows = db.execute(
                select(Message).where(Message.session_id == self.session_id).order_by(Message.created_at)
            ).scalars().all()
            return [self._Msg("human" if r.role == "user" else "ai", r.content, r.created_at) for r in rows]

    def add_user_message(self, text: str):
        with self._db_session_factory() as db:
            m = Message(session_id=self.session_id, role="user", content=text, created_at=datetime.datetime.utcnow())
            db.add(m)
            db.commit()

    def add_ai_message(self, text: str):
        with self._db_session_factory() as db:
            m = Message(session_id=self.session_id, role="ai", content=text, created_at=datetime.datetime.utcnow())
            db.add(m)
            db.commit()

    def clear(self):
        with self._db_session_factory() as db:
            db.execute(
                select(Message).where(Message.session_id == self.session_id).delete(synchronize_session=False)
            )
            db.commit()


# -----------------------
# Session metadata helpers (last_visit and doctor)
# -----------------------
class SessionStore:
    def __init__(self, db_session_factory):
        self._db_session_factory = db_session_factory

    def set_visit(self, session_id: str, doctor: str, visit_time: datetime.datetime = None):
        visit_time = visit_time or datetime.datetime.utcnow()
        with self._db_session_factory() as db:
            doc = db.execute(select(SessionMeta).where(SessionMeta.session_id == session_id)).scalars().first()
            if doc:
                doc.last_visit = visit_time
                doc.doctor = doctor
            else:
                db.add(SessionMeta(session_id=session_id, last_visit=visit_time, doctor=doctor))
            db.commit()

    def get_last_visit(self, session_id: str) -> Dict[str, Any]:
        with self._db_session_factory() as db:
            doc = db.execute(select(SessionMeta).where(SessionMeta.session_id == session_id)).scalars().first()
            if not doc:
                return {}
            return {"session_id": doc.session_id, "last_visit": doc.last_visit, "doctor": doc.doctor}


# -----------------------
# Setup DB engine and session factory
# -----------------------
# Replace with your Postgres URL:
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/langchain_chat_demo"

engine = create_engine(DATABASE_URL, echo=False, future=True)
Base.metadata.create_all(bind=engine)

# session factory
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


# -----------------------
# LangChain pipeline + RunnableWithMessageHistory wiring
# -----------------------
# Initialize LLM (Ollama local)
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

sys_prompt = "You are a helpful assistant called Follyb."

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(sys_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

pipeline = prompt_template | llm | StrOutputParser()

# get_session_history callable required by RunnableWithMessageHistory
def get_session_history(session_id: str):
    # return a PostgresChatMessageHistory for given session_id
    return PostgresChatMessageHistory(db_session_factory=lambda: SessionLocal(), session_id=session_id)


pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)

# session store for metadata
session_store = SessionStore(db_session_factory=lambda: SessionLocal())

# -----------------------
# Example application flow
# -----------------------
if __name__ == "__main__":
    # Example session id (use real user id / cookie / auth id in production)
    session_id = "patient_001"

    # 1) Simulate a patient first visit: they say hi and are assigned Dr. Ada
    print("=== First visit ===")
    resp1 = pipeline_with_history.invoke(
        {"query": "Hi, my name is Folly. I have a headache."},
        config={"configurable": {"session_id": session_id}}
    )
    print("Bot:", resp1)

    # Record doctor and visit in session metadata
    session_store.set_visit(session_id, doctor="Dr. Ada", visit_time=datetime.datetime.utcnow())

    # 2) Later in the same session: ask who attended them
    resp2 = pipeline_with_history.invoke(
        {"query": "Who attended to me during my last visit?"},
        config={"configurable": {"session_id": session_id}}
    )
    print("Bot:", resp2)

    # 3) Simulate user returning the next day (new process run also works because of persisted DB)
    print("\n=== Returning user (new interaction) ===")
    # Retrieve metadata to construct a prompt (you can also let LLM access stored message history)
    meta = session_store.get_last_visit(session_id)
    if meta:
        last_visit_str = meta["last_visit"].strftime("%Y-%m-%d %H:%M:%S") if meta["last_visit"] else "unknown"
        doctor = meta.get("doctor", "unknown")
        welcome = f"Welcome back! Your last visit was on {last_visit_str}, attended by {doctor}."
    else:
        welcome = "Welcome back! I don't see a previous visit."

    # Option A: provide welcome context to LLM by prepending as a query
    resp3 = pipeline_with_history.invoke(
        {"query": f"{welcome} How can I help you today?"},
        config={"configurable": {"session_id": session_id}}
    )
    print("Bot:", resp3)

    # Inspect raw persisted messages for debugging
    history_obj = get_session_history(session_id)
    print("\nPersisted Messages:")
    for m in history_obj.messages:
        print(f"- ({m.type}) {m.content}")
