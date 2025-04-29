import requests

# URL of our local app
url = 'http://127.0.0.1:8000/predict'

# Example payload
payload = {
    "reddit_comment": "This looks like a suspicious comment!"
}

# Send POST request
response = requests.post(url, json=payload)

# Print out the response
print(response.json())
