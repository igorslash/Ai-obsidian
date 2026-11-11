import os
from pathlib import Path
from typing import Literal, TypedDict

import Stemmer
from langgraph.graph import END, START, StateGraph
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from openai import OpenAI

# НАСТРОЙКИ LLM (подходит для OpenAI, OpenRouter, Ollama и т.д.) ---
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)



# ШАГ 1: Индексация (SimpleDirectoryReader + BM25Retriever)

NOTES_DIR = Path(__file__).resolve().parent / "notes"
NOTES_DIR.mkdir(parents=True, exist_ok=True)

documents = SimpleDirectoryReader(str(NOTES_DIR)).load_data()
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)

retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    stemmer=Stemmer.Stemmer("russian"),
    language="russian",
    similarity_top_k=min(3, len(nodes)),
)



# ШАГ 2: Состояние графа (State)

class State(TypedDict):
    question: str          # Вопрос пользователя
    documents: list[str]   # Найденные документы
    answer: str            # Финальный ответ


#Узлы графа (retrieve, grade_documents, generate)

def retrieve(state: State) -> dict:
    """Поиск документов через BM25Retriever."""
    retrieved = retriever.retrieve(state["question"])
    docs = [n.text for n in retrieved if (n.score or 0.0) > 0]
    return {"documents": docs}


def grade_documents(state: State) -> dict:
    """Оценка релевантности найденных документов вопросу."""
    question = state["question"]
    relevant_docs = []

    for doc in state["documents"]:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Оцени, помогает ли документ ответить на вопрос. Ответь одним словом: yes или no.",
                    },
                    {
                        "role": "user",
                        "content": f"Вопрос: {question}\n\nДокумент:\n{doc}",
                    },
                ],
                temperature=0,
            )
            verdict = resp.choices[0].message.content.strip().lower()
            if verdict.startswith(("yes", "да")):
                relevant_docs.append(doc)
        except Exception:
            # Если LLM недоступна, оставляем документ, найденный по BM25
            relevant_docs.append(doc)

    if not relevant_docs:
        return {
            "documents": [],
            "answer": "В заметках нет подходящей информации по вашему вопросу.",
        }

    return {"documents": relevant_docs}


def generate(state: State) -> dict:
    """Генерация финального ответа по отобранным документам."""
    context = "\n\n---\n\n".join(state["documents"])
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Отвечай на вопрос кратко и по делу, опираясь только на контекст.",
                },
                {
                    "role": "user",
                    "content": f"Контекст:\n{context}\n\nВопрос: {state['question']}",
                },
            ],
            temperature=0.2,
        )
        return {"answer": resp.choices[0].message.content.strip()}
    except Exception as e:
        return {"answer": f"[Не удалось вызвать LLM ({e})]\nНайденный контекст:\n{context}"}



# ШАГ 4: Сборка StateGraph с одним условным переходом

def route_after_grade(state: State) -> Literal["generate", "end"]:
    return "generate" if state["documents"] else "end"


workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grade,
    {
        "generate": "generate",
        "end": END,
    },
)
workflow.add_edge("generate", END)

app = workflow.compile()


# --- ЗАПУСК ---
if __name__ == "__main__":
    print("=== База знаний (LlamaIndex BM25 + LangGraph) ===")
    print("Положите файлы в ./notes. Для выхода введите 'exit' или пустую строку.\n")

    try:
        while True:
            user_question = input("Вопрос > ").strip()
            if not user_question or user_question.lower() in ("exit", "quit", "q", "выход"):
                break

            result = app.invoke({
                "question": user_question,
                "documents": [],
                "answer": "",
            })
            print(f"\nОтвет: {result['answer']}\n")
    except KeyboardInterrupt:
        print("\nЗавершение работы.")
