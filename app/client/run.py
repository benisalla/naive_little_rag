import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

host = os.environ["HOST"]
port = "1122"


# streamlit run run.py  host port
# streamlit run app/client/main.py --server.address localhost --server.port 1111
command = ['streamlit', 'run', 'app/client/main.py', '--server.address', host, '--server.port', port]
try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Command failed with error: {e}")
