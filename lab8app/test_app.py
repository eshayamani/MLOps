import requests

url = "http://127.0.0.1:8000/predict"

data = {"input_data": [0.03807591, 0.05068014, 0.06169621, 0.02187235, 0.02471326, 0.02972524, -0.01427858, -0.0036728, -0.03964116, -0.01430964]}

response = requests.post(url, json=data)

print("Response from the API:", response.json())
