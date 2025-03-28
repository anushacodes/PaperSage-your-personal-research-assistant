# PaperSage-your-personal-research-assistant
PaperSage is a lightweight AI assistant for answering questions from research papers. Users can upload a PDF, and the system extracts, embeds, and retrieves relevant content before generating answers using an LLM.

This project implements a **Retrieval-Augmented Generation (RAG)** system using **LangChain, ChromaDB, and Ollama** for querying PDF documents efficiently. It enables embedding text from PDFs, storing it in a vector database, and retrieving contextually relevant information using an **LLM (Mistral via Ollama)**.

## 🚀 Features

- **PDF Document Ingestion**: Automatically loads and processes PDFs from the `data/` directory.
- **Text Chunking**: Splits PDFs into smaller chunks for optimized vector search.
- **Vector Database Storage**: Uses **ChromaDB** to persist embeddings for fast retrieval.
- **Ollama Embeddings**: Uses `nomic-embed-text` for high-quality text embeddings.
- **Query System**: Users can search the database using natural language queries.
- **LLM-Powered Answering**: Uses the **Mistral** model via **Ollama** to generate answers based on retrieved documents.
- **CLI Commands**:
  - Reset the database (`--reset`)
  - Populate the database from PDFs
  - Query the stored data (`--query "your question"`)
- **Web Interface**: Simple Flask-based UI for querying the system.

## 📂 Project Structure

```
📁 root/
├── 📁 data/                # Directory for storing PDFs
├── 📁 chroma/              # ChromaDB storage
├── 📁 frontend/            # Frontend folder containing HTML & CSS
│   ├── 📄 index.html       # Web interface
│   ├── 📄 style.css        # Styling for the web interface
├── 📁 templates/           # Flask template folder
│   ├── 📄 index.html       # Flask-compatible HTML template
├── 📁 static/              # Flask static folder
│   ├── 📄 style.css        # CSS for Flask app
├── 📄 embed.py             # Script for processing and embedding documents
├── 📄 query.py             # Script for querying the database
├── 📄 app.py               # Flask web server
├── 📄 requirements.txt      # Dependencies
├── 📄 README.md            # Project documentation
```

## 🔧 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/anushacodes/pdf-rag-system.git  # CORRECT THIS
   cd pdf-rag-system
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Ollama (if not already installed)**
   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

## ⚙️ Usage

### 1️⃣ Populate the Database

Place your PDFs inside the `data/` folder and run:

```bash
python embed.py
```

### 2️⃣ Query the Database (CLI)

Ask a question based on stored documents:

```bash
python query.py --query "Describe the attention mechanism"
```

### 3️⃣ Interactive CLI Mode

Start an interactive session for multiple queries:

```bash
python query.py --interactive
```

### 4️⃣ Start the Web UI

Run the Flask web server to query via a browser:

```bash
python app.py
```

Visit `http://127.0.0.1:5001/` in your browser.

### 5️⃣ Reset the Database

To clear stored data, run:

```bash
python embed.py --reset
```

## 📌 Future Enhancements

- Multi-file format support (TXT, DOCX, etc.)
- Support for multiple LLM models


