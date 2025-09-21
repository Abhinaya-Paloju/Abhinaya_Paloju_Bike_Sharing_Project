import requests

url = "http://127.0.0.1:8000/predict"

# Three sample scenarios
samples = {
    "Normal day": {
        "season": 1,
        "holiday": 0,
        "workingday": 1,
        "weather": 1,
        "temp": 0.24,
        "atemp": 0.2879,
        "humidity": 0.81,
        "windspeed": 0.0
    },
    "Holiday": {
        "season": 1,
        "holiday": 1,
        "workingday": 0,
        "weather": 1,
        "temp": 0.28,
        "atemp": 0.32,
        "humidity": 0.75,
        "windspeed": 0.1
    },
    "Stormy day": {
        "season": 1,
        "holiday": 0,
        "workingday": 1,
        "weather": 3,
        "temp": 0.15,
        "atemp": 0.12,
        "humidity": 0.95,
        "windspeed": 0.4
    }
}

for name, sample in samples.items():
    try:
        response = requests.post(url, json=sample)
        response.raise_for_status()
        data = response.json()
        print(f"{name}: Predicted count = {data.get('predicted_cnt', data.get('error'))}")
    except Exception as e:
        print(f"{name}: Error = {e}")
