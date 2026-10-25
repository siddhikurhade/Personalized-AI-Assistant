
# Personalized AI Assistant using Local LLM and RAG

An AI assistant that uses a locally running language model, Retrieval-Augmented Generation (RAG), and document embeddings to answer questions.

## Features

- Local LLM-based question answering
- Document-based question answering using RAG
- FAISS vector search
- Sentence Transformers embeddings
- Streamlit user interface
- Conversation history using SQLite
- FastAPI endpoint

## Tech Stack

- Python
- Streamlit
- FastAPI
- Hugging Face Transformers
- PyTorch
- Sentence Transformers
- FAISS
- SQLite

## Project Structure

```text
personalized_ai_assistant/
├── app.py
├── api.py
├── config.py
├── database.py
├── llm.py
├── rag.py
├── prompts.py
├── requirements.txt
├── data/
├── evaluation/
└── training/
    └── train_lora.py
```

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies:

   `pip install -r requirements.txt`

4. Configure the model and other settings in a local `.env` file.
5. Start the application:

   `python -m streamlit run app.py`

## Local Model

This project is designed to run a language model locally. The required model files may be downloaded separately on first use, depending on the configuration.

## Note

Model performance depends on available RAM, CPU/GPU hardware, and model size.
