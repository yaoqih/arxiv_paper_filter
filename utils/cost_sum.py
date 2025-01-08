import requests
import json

url = "https://api.chatanywhere.tech/v1/query/usage_details"

payload = json.dumps({
   "model": "gpt-3.5-turbo%",
   "hours": 24
})
headers = {
   'Authorization': 'sk-xxxxxxxx',
   'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)