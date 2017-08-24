import requests
res = requests.post('http://54.200.82.138:5000/model', json={"gen_length":150})

if res.ok:
    print (res.json())
