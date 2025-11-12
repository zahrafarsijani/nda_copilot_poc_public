# NDA Review Copilot

A retrieval-augmented GenAI prototype that accelerates NDA review by comparing uploaded contracts against a Market-Standard Playbook.  
Built with Streamlit, Azure OpenAI, and LangChain, this prototype demonstrates clause-level analysis, prompt-injection defense, and retrieval-augmented grounding.

---

## Overview

The NDA Review Copilot allows users to:
- Upload an NDA in `.docx` format.  
- Compare it against a Market-Standard Playbook for deviations and risks.  
- Receive a structured JSON report and a concise lawyer-style summary.  
- Interact with an optional chat interface for follow-up questions.  

---

## Features

- Clause-level contract review using Azure OpenAI (GPT-4o).  
- Retrieval-Augmented Generation (RAG-lite) with FAISS vector store.  
- Prompt-injection detection and document classification safeguards.  
- Structured JSON output and text summaries.  
- Run logs capturing file name, model, runtime, and security events.  

---

## Tech Stack

| Layer | Technology | Purpose |
|--------|-------------|----------|
| Frontend | Streamlit | User interface for upload, analysis, and chat |
| LLM Engine | Azure OpenAI (GPT-4.1) | Clause-level reasoning and report generation |
| Retrieval Layer | LangChain, FAISS, Azure Embeddings | Retrieve relevant Playbook sections |
| Data | Python-docx | Extracts text from uploaded NDAs |
| Security | Regex + GPT-based classification | Detect prompt injection and non-NDA uploads |
| Environment | Python 3.12 + virtual environment | Local development and isolation |

---

## Prerequisites

- Python 3.12.9  
- Azure OpenAI resource with:
  - A chat model deployment (e.g., `gpt-4.1`)
  - An embedding model deployment (e.g., `text-embedding-3-large`)
- Files required:
  - `NDA Playbook.docx`
  - `app.py`
  - `setup.sh`, `run.sh`
  - `requirements.txt`
  - `prompts/nda_copilot_<version>.txt`

---

## Environment Setup

### Step 1: Create `.env` File

Create a file named `.env` in the project root and add:

```
AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
AZURE_OPENAI_API_KEY="<your-api-key>"
AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
RETRIEVAL_TOP_K=3
```

---

### Step 2: Install and Configure Environment

Run:

```
bash setup.sh
```

This script:
- Detects or installs Python 3.12  
- Creates `.venv`  
- Installs all dependencies from `requirements.txt`

---

### Step 3: Launch the Application

Run:

```
bash run.sh
```

This will:
- Activate the virtual environment  
- Load `.env` variables  
- Launch the Streamlit app at `http://localhost:8501`

---

## Repository Structure

```
.
├── app.py
├── setup.sh
├── run.sh
├── requirements.txt
├── .env.example
├── NDA Playbook.docx
├── prompts/
│   └── nda_copilot_v1.5-secure-rag.txt
└── faiss_index/
```

---

## Notes

- The prototype is for demonstration purposes only and does not provide legal advice.  
- Ensure `.env` contains valid Azure credentials before running.  
- All runs and security events are logged to `run_log.csv` and `security_log.csv`.  
