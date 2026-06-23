# CRAG - Corrective Retrieval Augmented Generation

A production-style AI system that retrieves documents, grades their relevance, and falls back to web search when local knowledge is insufficient.

## Architecture

User Query → Retrieve → Grade → Decision → Generate → Final Answer
                                    ↓              ↑
                              Web Search ──────────┘

## Tech Stack

- **LangGraph** - Graph-based AI workflow
- **FAISS** - Vector similarity search
- **Groq (LLaMA 3.1)** - LLM for grading and generation
- **Tavily API** - Web search fallback
- **HuggingFace Embeddings** - Text to vector conversion

## How it works

1. **Retrieve** - FAISS searches local documents for similar content
2. **Grade** - LLM checks if retrieved docs are actually relevant
3. **Decide** - If relevant, use local docs. If not, trigger web search
4. **Generate** - Final answer generated from best available context

## Setup

1. Clone the repo
2. Create virtual environment
3. Install dependencies
4. Add API keys in .env file
5. Run the app

```bash
pip install -r requirements.txt
python crag_app.py
```

## Environment Variables

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```
## Screenshots

### Graph Architecture
<img width="367" height="433" alt="WhatsApp Image 2026-06-23 at 5 10 39 PM" src="https://github.com/user-attachments/assets/000dc441-b30d-4d7e-8478-234b32375c92" />
<img width="739" height="793" alt="WhatsApp Image 2026-06-23 at 6 09 49 PM" src="https://github.com/user-attachments/assets/f0f3ae1d-560d-44d4-9231-dbb23fa5dfeb" />
