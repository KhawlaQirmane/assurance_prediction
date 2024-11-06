import requests

url = "http://localhost:5000/predict"
data = {
    "age": 30,
    "sex": "male",
    "bmi": 25.3,
    "children": 2,
    "smoker": "no",
    "region": "northeast"
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
