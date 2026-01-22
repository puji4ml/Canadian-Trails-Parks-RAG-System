# Canadian Trails & Parks RAG System

An end-to-end **MLOps-focused Retrieval Augmented Generation (RAG) system** built on Canadian trails and parks data. The project is designed to **compare, evaluate, and optimize multiple RAG pipelines** across chunking strategies, embedding models, and retrieval setups instead of relying on guesswork.

The goal is simple: **know which RAG configuration works best for your data, accuracy, relevance, and cost**.

---

## 🧠 Architecture

**Full Stack RAG Optimizer**

1. **Frontend**

   * Streamlit application
   * Upload documents and ask questions

2. **Backend**

   * Parallel RAG pipelines with different configurations
   * FastAPI-style modular design (logic separated for scalability)

3. **AI Core**

   * Multiple chunking strategies
   * Multiple embedding models
   * LLM-based evaluator scoring responses on:

     * Accuracy
     * Relevance
     * Cost efficiency

4. **Data & Infra**

   * Vector databases per configuration
   * Evaluation dashboards
   * Docker-ready for deployment

---

## 📂 Repository Structure

```
├── app.py                     # Streamlit application
├── pipeline.py                # Core RAG pipeline logic
├── evaluation.ipynb           # RAG evaluation and comparison
├── dataingestion.ipynb        # Chunking and vector DB creation
├── scripts/                   # Data collection scripts
│   ├── collect_canada_trails_fixed.py
│   ├── collect_parks_canada_enhanced.py
│   └── combine_canada_data.py
├── data/
│   ├── raw/                   # Raw collected datasets
│   ├── processed/             # Train/test datasets
│   └── chunked/               # Chunked documents
├── requirements.txt
└── README.md
```

Large artifacts such as embeddings, vector databases, and binaries are intentionally excluded from Git history.

---

## 🛠️ Environment Setup

### 1️⃣ Initialize Project

```bash
uv init
uv venv
```

### 2️⃣ Install Dependencies

```bash
uv add -r requirements.txt
uv add ipykernel
```

---

## 📊 Phase 1: Data Ingestion Pipeline

### Data Collection

Run scripts in the following order:

```bash
python scripts/collect_canada_trails_fixed.py
python scripts/collect_parks_canada_enhanced.py
python scripts/combine_canada_data.py
```

Final datasets will be available in:

```
data/processed/
```

---

### Document Chunking & Vector DB

* Open `dataingestion.ipynb`
* Run all cells sequentially
* This will:

  * Chunk documents
  * Build multiple vector databases
  * Perform basic retrieval validation

---

## 🔍 Phase 2: Query Processing & Evaluation

### Free Local LLM Setup (Ollama)

Install Ollama:
[https://ollama.com/download](https://ollama.com/download)

Start the service:

```bash
ollama serve
```

Pull models:

```bash
ollama pull llama3.2:3b
ollama pull llama3.1:8b
```

Verify:

```bash
ollama list
```

Test:

```bash
ollama run llama3.2:3b "What are the best hiking trails in Canada?"
```

---

### Evaluation

* Run `pipeline.ipynb`
* Execute `evaluation.ipynb`
* Compare RAG outputs across:

  * Chunk sizes
  * Embedding models
  * Retrieval strategies

---

## 🖥️ Running the App

Set environment variable:

```powershell
$env:GROQ_API_KEY="gsk_******"
```

Run Streamlit:

```bash
uv run streamlit run app.py
```

---
