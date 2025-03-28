import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Constants
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context. Say you don't know the answer if the context does not match the question:

{context}

---

Answer the question based on the above context: {question}
"""

def get_embedding_function():
    """Returns an embedding function using Ollama's nomic-embed-text model."""
    return OllamaEmbeddings(model="nomic-embed-text")

def query_rag(query_text: str):
    """Queries the ChromaDB for relevant text chunks and generates an answer using LLM."""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    cleaned_response = response_text.strip()
    print(cleaned_response)
    return cleaned_response

def interactive_mode():
    """Runs an interactive CLI mode for continuous queries."""
    print("💬 Interactive RAG Query Mode (type 'exit' to quit)")
    while True:
        query_text = input("Enter your query: ")
        if query_text.lower() == "exit":
            print("Exiting interactive mode.")
            break
        query_rag(query_text)

def main():
    """Handles command-line arguments for querying the database."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Query text for RAG.")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode.")
    args = parser.parse_args()

    if args.query:
        query_rag(args.query)
    elif args.interactive:
        interactive_mode()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
