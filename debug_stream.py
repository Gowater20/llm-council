
import requests
import json
import sseclient

url = "http://localhost:8001/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "council-llm",
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": True  # Cursor uses streaming by default
}

print(f"Connecting to {url}...")
try:
    response = requests.post(url, headers=headers, json=data, stream=True)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Error: {response.text}")
        exit(1)

    print("Listening for stream...")
    client = sseclient.SSEClient(response)
    for event in client.events():
        print(f"Event: {event.data}")
        if event.data == "[DONE]":
            break
            
except Exception as e:
    print(f"Exception: {e}")
