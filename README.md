# RAG-Based Knowledge Assistant (Groq)

A Retrieval-Augmented Generation (RAG) based knowledge assistant that answers user queries using custom documents by combining **LangChain**, **FAISS**, and **Groq LLMs**.

---

## 🚀 Features
- Document-based question answering
- Semantic search using FAISS vector store
- Fast inference using Groq LLM API
- Context-aware responses (RAG architecture)
- Secure API key handling with `.env`

---

## 🧠 How It Works
1. Documents are loaded and split into chunks  
2. Chunks are embedded and stored in **FAISS**
3. User query is embedded and matched with relevant chunks
4. Retrieved context + query is sent to **Groq LLM**
5. Accurate, context-grounded answer is generated

---

## 🛠️ Tech Stack
- Python
- LangChain
- FAISS
- Groq LLM API
- python-dotenv

---
