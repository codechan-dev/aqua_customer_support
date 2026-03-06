from pathlib import Path

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "watercan_docs.txt"


def load_documents() -> list[str]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Sample document not found at {DATA_PATH}")

    text = DATA_PATH.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    return chunks


def build_vector_store(chunks: list[str]) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store


def answer_question(retriever, question: str) -> str:
    docs = retriever.invoke(question)
    if not docs:
        return "I could not find information about that in the watercan delivery documents."

    context_parts = []
    for i, doc in enumerate(docs, start=1):
        context_parts.append(f"Snippet {i}:\n{doc.page_content}")

    context_text = "\n\n".join(context_parts)

    return (
        "Here is what I found in the WaterCan delivery documentation related to your question:\n\n"
        f"{context_text}\n\n"
        "The answer above is taken directly from the internal docs. "
        "If this does not fully answer your question, you may need to contact support or check additional documentation."
    )


def main():
    load_dotenv()

    print("Loading sample watercan delivery documents...")
    chunks = load_documents()

    print(f"Loaded {len(chunks)} text chunks. Building vector store with free HuggingFace embeddings...")
    vector_store = build_vector_store(chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    print("\nWaterCan Delivery RAG assistant is ready (retrieval-only, no external LLM or paid APIs).")
    print("Ask questions about the watercan delivery system (type 'exit' to quit).")

    while True:
        question = input("\nYour question: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            answer = answer_question(retriever, question)
            print("\nAnswer:")
            print(answer)
        except Exception as e:
            print(f"Error while generating answer: {e}")


if __name__ == "__main__":
    main()

