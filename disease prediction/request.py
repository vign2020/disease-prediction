import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age':2, 'sex':9, 'cp':6})

print(r.json())