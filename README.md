# WaterCan Delivery RAG Assistant

A **Retrieval Augmented Generation (RAG)** chatbot for the WaterCan delivery system. It answers questions using sample documentation, free HuggingFace embeddings, FAISS vector search, and a local Qwen model via Ollama—no paid APIs required.

## Features

- **RAG pipeline**: Load docs → chunk → embed → store in FAISS → retrieve on query → generate answer with Qwen
- **Free embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Local LLM**: Qwen 2.5 3B via Ollama (runs entirely on your machine)
- **Streamlit chat UI**: Conversation-style interface with response time tracking (seconds per answer)
- **Relevance check**: If the question is not related to the docs, the app returns "no relevant answer" instead of guessing

## Project structure

```
lanchanin-rag/
├── data/
│   └── watercan_docs.txt    # Sample WaterCan delivery documentation
├── rag_watercan.py          # Core RAG: load docs, chunk, embed, FAISS, retriever
├── streamlit_app.py         # Chat UI + Qwen (Ollama) + latency tracking
├── requirements.txt
└── README.md
```

## Tech stack

| Component        | Technology |
|----------------|------------|
| Framework       | LangChain |
| Text splitter   | `langchain-text-splitters` (RecursiveCharacterTextSplitter) |
| Embeddings      | `langchain-huggingface` (all-MiniLM-L6-v2) |
| Vector store    | FAISS (`langchain-community`) |
| LLM             | Qwen 2.5 3B via Ollama (`langchain-ollama`) |
| UI              | Streamlit |

## Prerequisites

1. **Python 3.10 or 3.11** (recommended; 3.14 may show Pydantic warnings)
2. **Ollama** installed and running, with Qwen pulled:
   ```bash
   ollama pull qwen2.5:3b
   ```

## Setup

1. Clone the repo and go to the project folder:
   ```bash
   cd lanchanin-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows (PowerShell - if scripts are allowed):
   .venv\Scripts\activate
   # Windows (Command Prompt):
   .venv\Scripts\activate.bat
   # Or run Python directly: .venv\Scripts\python.exe
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

5. Open the URL in your browser (e.g. `http://localhost:8501`) and start asking questions about the WaterCan delivery system.

## Example questions

- What is the minimum order quantity?
- What are the subscription plans?
- How long does delivery take?
- Do I have to return the empty can?
- What payment methods do you accept?
- What is the refund policy?

## Optional: CLI mode

You can also run the retrieval-only flow from the terminal (no Streamlit, no Ollama):

```bash
python rag_watercan.py
```

## License

Use and modify as needed for learning or internal use.
