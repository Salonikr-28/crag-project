from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv
import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=key)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

sentences = [
    "Refund policy is 30 days after purchase.",
    "You can return products within one month.",
    "Our office is open Monday to Friday.",
    "We accept credit cards and UPI payments.",
]

db = FAISS.from_texts(sentences, embeddings)


def retrieve(query):
    results = db.similarity_search(query, k=2)
    docs = [r.page_content for r in results]
    return docs

def grade(query, doc):
    prompt = f"""You are a grader checking document relevance.
Question: {query}
Document: {doc}

Is this document relevant to answering the question? Reply with only one word: yes or no."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip().lower()
    return "yes" in answer

from tavily import TavilyClient

tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key)


def web_search(query):
    response = tavily_client.search(query=query, max_results=2)
    results = response["results"]
    snippets = [r["content"] for r in results]
    return snippets

def generate_answer(query, context_docs):
    context = "\n".join(context_docs)
    prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {query}

Answer in 1-2 sentences."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

class GraphState(TypedDict):
    query: str
    docs: List[str]
    relevant_docs: List[str]
    final_context: List[str]
    answer: str


def retrieve_node(state):
    docs = retrieve(state["query"])
    return {"docs": docs}


def grade_node(state):
    relevant = []
    for d in state["docs"]:
        if grade(state["query"], d):
            relevant.append(d)
    return {"relevant_docs": relevant}


def decide_node(state):
    if state["relevant_docs"]:
        return "generate"
    else:
        return "web_search"


def web_search_node(state):
    results = web_search(state["query"])
    return {"final_context": results}


def use_local_node(state):
    return {"final_context": state["relevant_docs"]}


def generate_node(state):
    answer = generate_answer(state["query"], state["final_context"])
    return {"answer": answer}

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("use_local", use_local_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    decide_node,
    {
        "generate": "use_local",
        "web_search": "web_search"
    }
)

workflow.add_edge("use_local", "generate")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

if __name__ == "__main__":
    result = app.invoke({"query": "What is the return policy?"})
    print("Query:", result["query"])
    print("Final Answer:", result["answer"])