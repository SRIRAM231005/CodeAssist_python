# CodeAssist

## Objective

CodeAssist is an intelligent coding assistant designed to provide clear, detailed explanations for specific functions, modules, or entire files within a GitHub repository. Given a repository link, file path, and a natural language query, CodeAssist analyzes the source code structure and retrieves only the most relevant context to generate accurate, noise-free explanations that stay grounded in the actual code.

Unlike a generic prompt-based setup, CodeAssist is built as a structured system that combines static code analysis, semantic retrieval, and controlled generation to minimize hallucinations and maximize relevance.

---

## Demo

<!-- Add demo video here -->

> 📹 *Demo video showing end-to-end usage of CodeAssist (query → retrieval → explanation)*

---

## Technologies Used

* **CodeBERT** — shared embedding model for:

  * Offline semantic indexing of source code
  * Online embedding of user queries for retrieval

* **Gemini 2.5 Flash** — used in two distinct stages:

  * Structured intent extraction (JSON-based)
  * Grounded explanation generation (context-constrained)

* **CodeSearchNet (Python subset)** — ~20K samples selected from ~455K total code snippets

* **ChromaDB** — vector database for similarity search

* **Python AST** — function and method-level code extraction

* **Streamlit** — interactive web interface

---

## System Overview

<!-- Architecture diagram will be added here -->

CodeAssist operates in two major phases: an **offline indexing phase** and an **online query & explanation phase**.

### 1. Offline Indexing (Embedding Pipeline)

A subset of approximately **20,000 Python code samples** is selected from the larger **~455,000 sample CodeSearchNet dataset**. The selection focuses on high-quality, well-structured code suitable for semantic retrieval.

Each code sample is processed and embedded using **CodeBERT**, producing vector representations that capture both syntactic structure and semantic intent. Along with embeddings, relevant metadata (such as source, identifiers, and contextual information) is stored.

These embeddings and metadata are persisted in **ChromaDB**, forming the semantic knowledge base used during query time. This stage is executed once and does not run during user interaction.

---

### 2. Online Query Processing & Explanation

When a user submits a query along with a GitHub repository link and file path, CodeAssist performs the following steps:

* **Code Retrieval & Parsing**
  The specified file is fetched from GitHub and parsed using Python’s **AST module** to extract individual functions and methods into a structured list.

* **Intent Extraction**
  The user query is passed to **Gemini 2.5 Flash**, which extracts:

  * Referenced function or method names
  * The structural intent of the query
    This information is returned in a strictly defined JSON format.

* **Function Filtering**
  Based on the extracted intent, only the relevant functions are selected from the AST-extracted list, reducing unnecessary context.

* **Semantic Retrieval**
  The raw user query is embedded using **CodeBERT** and used to perform similarity search over **ChromaDB**, retrieving the most relevant documents and metadata from the offline index.

* **Grounded Explanation Generation**
  The filtered functions, retrieved metadata, and structured intent are provided to **Gemini 2.5 Flash**, which generates a precise explanation strictly grounded in the supplied context.

---

## Project Structure

```text
CodeAssist/
├── main.py                 # Generates embeddings for ~20K CodeSearchNet samples
├── chromadb_setup.py       # Initializes ChromaDB and stores embeddings + metadata
├── app/
│   └── streamlit_app.py    # Streamlit application (query, retrieval, explanation)
├── requirements.txt
```

> ⚠️ Note: The generated embeddings and ChromaDB storage directory are **not included** in the repository due to their large size.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CodeAssist
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Embeddings (Offline Step)

Run the following script to embed the selected CodeSearchNet samples and store them in ChromaDB:

```bash
python main.py
python chromadb_setup.py
```

> This step may take time depending on system resources. It needs to be executed only once.

### 5. Run the Application

```bash
streamlit run app/streamlit_app.py
```

---

## Notes

* CodeAssist is designed to work with **real repositories and real code**, not synthetic examples.
* The system prioritizes **relevance and grounding**, avoiding unnecessary context in explanations.
* All explanations are generated using retrieved code and metadata, minimizing hallucinations.

---

## Future Improvements

* Support for additional programming languages
* Enhanced metadata filtering strategies
* Optional persistence of user query history
