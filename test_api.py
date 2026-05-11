import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "SAP_%": 0.2,
    "W_C_Ratio": 0.38,
    "Weight_g": 2500,
    "Area_mm2": 10000,
    "Slump_mm": 70,
    "Age_days": 28
}

response = requests.post(url, json=data)

print("Response:")
print(response.json())