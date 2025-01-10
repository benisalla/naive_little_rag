import requests
import os
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        self.api_url = str(os.getenv("API_URL")).strip()
        self.token = str(os.getenv("TOKEN")).strip()
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def query(self, payload):
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()