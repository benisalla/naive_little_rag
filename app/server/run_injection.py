import os
from tkinter import SEPARATOR
import chromadb
import tiktoken
from dotenv import load_dotenv
from app.client.utils.constant import BACKUP_SEPARATORS, CHUNK_OVERLAP, CHUNK_SIZE
from util import textDB2EmbDB, RagEmbedder
from transformers import AutoTokenizer
from llama_index.core.node_parser.text.token import TokenTextSplitter

load_dotenv()

# Read the .env file
docs_dir = os.environ['DOCUMENTS_DIR']
collection_name = os.environ['COLLECTION_NAME']
emb_model_name = os.environ["EMB_MODEL_NAME"]
tokens_per_chunk = int(os.environ["TOKENS_PER_CHUNK"])
chroma_db_path = os.environ["CHROMA_DATABASE_PATH"]
host = os.environ["HOST"]
port = os.environ["PORT"]
hf_token = os.getenv('HF_TOKEN')
tok_model_name = os.getenv('TOK_MODEL_NAME')

print("Environment variables loaded.")

# sentence transformers token text splitter
token_splitter = TokenTextSplitter(
    separator=SEPARATOR,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    backup_separators=BACKUP_SEPARATORS,
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    # tokenizer = AutoTokenizer.from_pretrained(tok_model_name, use_auth_token=hf_token).encode
    # tokenizer = AutoTokenizer.from_pretrained(tok_model_name, token=hf_token).encode
)

print("Token text splitter configured.")

# embedding model
emb_fun = RagEmbedder(emb_model_name)
print("Embedding model initialized.")

print("Connecting to Chroma database...")
chroma_client = chromadb.PersistentClient(path=chroma_db_path)
try:
    chroma_client.delete_collection(collection_name)
    print("Existing collection deleted.")
except Exception as e:
    print(f"Failed to delete existing collection: {e}")

chroma_collection = chroma_client.create_collection(collection_name, embedding_function=emb_fun)
print("New collection created.")

# Create the docs_dir if it does not exist
if not os.path.exists(docs_dir):
    os.makedirs(docs_dir)
    print("Document directory created:", docs_dir)

# embedding documents and store them in our collection
print("Starting the embedding of documents...")
chroma_collection = textDB2EmbDB(chroma_collection, docs_dir, token_splitter, is_add=False)
print("Documents embedded and stored.")

print("Chroma database setup complete.")
print(f"Total chunks in the collection: {chroma_collection.count()}")
