# Evidence Validator – Multi Agentic RAG

Multi-agent Retrieval-Augmented Generation (RAG) system that evaluates factual or scientific claims against uploaded source documents using structured reasoning and Natural Language Inference (NLI).

## What It Does

Upload PDFs, ask questions, and receive evidence-backed validated answers.  
Each generated claim is checked against retrieved spans and labeled:

- **SUPPORTED** — Evidence confirms the claim  
- **CONTESTED** — Evidence contradicts the claim  
- **UNCERTAIN** — Evidence is inconclusive or mixed  

## How It Works

Query → Hybrid Retrieval → Claim Synthesis → NLI Validation → Final Answer


### Agents

1. **Claim Synthesizer** – Generates grounded, testable claims from retrieved evidence  
2. **Validator** – Uses DistilBERT-MNLI to evaluate whether evidence supports or contradicts each claim  
3. **Orchestrator** – Coordinates retrieval, synthesis, validation, and answer generation  

## Stack

- Retrieval: FAISS + BM25 + MMR  
- LLM: Claude 3.5 Haiku  
- Validation: DistilBERT-MNLI  
- Embeddings: BGE-base-en-v1.5  
- Backend: FastAPI  
- Frontend: Streamlit  

## Setup

```bash
git clone https://github.com/PANDEAD/Evidence-validator-Agentic-RAG.git
cd Evidence-validator-Agentic-RAG

python3.11 -m venv ragenv
source ragenv/bin/activate
pip install -r requirements.txt

# Place PDFs in data/pdfs/
python -m src.services.ingestion

# Add API Key

cp .env.example .env
 #then edit .env
ANTHROPIC_API_KEY=your_key_here

#run the backend
python app.py

#run the frontend in a new terminal
streamlit run ui/app.py
```


