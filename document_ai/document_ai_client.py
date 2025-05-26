import base64
import time
import requests
import os
from dotenv import load_dotenv
load_dotenv()
import google.auth
import google.auth.transport.requests

# Configuration for Google Document AI
PROJECT_ID = "629159515213"
LOCATION = "us"
PROCESSOR_ID = "425f4be70b86af78"
ENDPOINT_URL = f"https://{LOCATION}-documentai.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}:process"

def get_access_token():
    """Get access token using Application Default Credentials"""
    credentials, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token


def process_document(file_content, mime_type):
    """Process a document using Google Document AI REST API"""
    try:
        access_token = get_access_token()
        if not access_token:
            return {"error": "Could not retrieve access token"}

        # Encode document content
        encoded_content = base64.b64encode(file_content).decode("utf-8")
        payload = {
            "rawDocument": {
                "content": encoded_content,
                "mimeType": mime_type
            }
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        start_time = time.time()
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
        processing_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            return {"result": result, "processing_time": processing_time}
        else:
            return {"error": f"API Error {response.status_code}: {response.text}", "processing_time": processing_time}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"} 