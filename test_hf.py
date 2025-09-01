import requests

url = "http://127.0.0.1:8000/recommend"
payload = {
    "occasion": "Birthday",
    "age": 20,
    "gender": "female",
    "interests": "music,books",
    "budget_min": 500,
    "budget_max": 2000
}

response = requests.post(url, json=payload, timeout=60)

print("Status:", response.status_code)
print("Response:", response.json())
