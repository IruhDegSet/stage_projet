import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Cette ligne lève une exception pour les erreurs HTTP
    return response.json()

context = ("Explain how the YOLO method works")
prompt = context + " Can you please let us know more details about your recent purchase?"

try:
    data = query({"inputs": prompt})
    print("Success: The query was processed successfully!")
    print(data)
except requests.exceptions.HTTPError as err:
    print(f"HTTP error occurred: {err}")
except Exception as err:
    print(f"An error occurred: {err}")
