# Value Little Rag Demo Repository

This project demonstrates my capability to implement a simple RAG (Retrieval-Augmented Generation) system from scratch using Chroma and open-source models.

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/juridai-demo.git
cd valeo_little_rag

# Create and activate a Python environment
python3.11 -m venv venv
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

- Rename the file `.env.example` to `.env`.
- Add your Hugging Face token in the `.env` file.

## Running the Server (RAG Injection)

To set up the server and inject data for retrieval:

```bash
# Launch ChromaDB Server (in terminal 1)
python app/server/run_chromadb.py

# Embed and store data via injection (in terminal 2)
python app/server/run_injection.py
```

## Running the Client (RAG Retrieval)

To test the retrieval functionality:

```bash
# Run the RAG retrieval system (in terminal 3)
python app/client/run.py
```