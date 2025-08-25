# ProteinScope — Nutrition RAG + Agent (FAISS + Ollama)

ProteinScope is a focused Retrieval‑Augmented Generation (RAG) project for **dietary protein**:
- Scrapes **Reddit**, **PubMed**, and **protein blogs/news** (incremental; de‑duped; append-only)
- Filters & chunks to a clean **corpus**
- Builds a **FAISS** vector index with sentence‑transformer embeddings
- Answers questions using **Ollama** (default: `llama3.1:8b`) with citations & guardrails

---

## 1) Quick Start

### A) Prereqs
- **Python 3.10+** (3.11/3.12 OK; 3.13 works but be up-to-date on wheels)
- **Ollama** (for the local LLM):
  ```bash
  # Install Ollama from https://ollama.ai
  ollama pull llama3.1:8b
  # Make sure the server is running (it usually is):
  ollama serve