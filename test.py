import json
import urllib.request

url = "https://data-poisoning-detection-api.onrender.com/analyze"

payload = {
    "values": [
        [45, 22],
        [50, 24],
        [52, 25],
        [500, 100],
        [45, 22],
        [50, 24],
        [52, 25],
        [500, 100],
        [80, 34],
        [59, 43],
        [59, 89],
        [5000, 1000]
    ],
    "methods": ["z_score", "isolation_forest", "lof", "hybrid"]
}

data = json.dumps(payload).encode("utf-8")

req = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(req) as response:
    result = json.loads(response.read().decode("utf-8"))
    print(json.dumps(result, indent=2))