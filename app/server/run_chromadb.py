import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

# Read the .env file
chroma_db_path = os.environ["CHROMA_DATABASE_PATH"]
host = os.environ["HOST"]
port = os.environ["PORT"]

# chroma run --path ../chroma_db --host localhost --port 9999
command = ['chroma', 'run', '--path', chroma_db_path, '--host', host, '--port', port]
subprocess.run(command)
