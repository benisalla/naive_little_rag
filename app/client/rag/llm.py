import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

class LLMClient:
    def __init__(self):
        self.api_url = str(os.getenv("API_URL")).strip()
        self.hf_token = str(os.getenv("HF_TOKEN")).strip()
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
    
    def query(self, payload):
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_message = response.text
            try:
                # Attempt to parse JSON error message
                error_data = json.loads(error_message)
                print("Error from API:", error_data)
            except json.JSONDecodeError:
                print("Non-JSON error from API:", error_message)
            raise
        return response.json()