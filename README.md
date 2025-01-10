# Value Little Rag Demo Repository

Part of my projects to demonstrate my capability to implement a simple rag from scratch with Chroma and open-source models.

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/juridai-demo.git
cd valeo_little_rag

# Create and activate Python environment
python3.11 -m venv venv
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Server (rag injection)

```bash
# Launch ChromaDB Server (in terminal 1)
python app/server/chroma.py

# Run injection to embed and store data (in terminal 2)
python app/server/injection.py
```

## Running the Client (rag Retrieval)

Run just retriever

```bash
# Run rag system (in terminal 3)
python app/client/retriever.py
```